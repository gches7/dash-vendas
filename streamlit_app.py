"""Dashboard Executivo de Vendas — arquivo único (Streamlit + Plotly).

Execução:   streamlit run dashboard.py
Dados:      coloque o CSV de pedidos na pasta `data/` ao lado deste arquivo
            (ou envie o arquivo em "Fonte de dados", na barra lateral).
Dependências: streamlit, pandas, numpy, plotly, requests (ver requirements.txt).

Seções do arquivo: 1) formatação · 2) leitura/limpeza · 3) métricas, projeção e backtest ·
4) gráficos · 5) mapa das regiões · 6) interface (4 abas: Visão Geral, Distribuição Geográfica,
Categorias e Canais, Sazonalidade e Planejamento).
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots


# ============================================================================
# 1. FORMATAÇÃO (pt-BR)
# ============================================================================
MESES_ABREV = ["jan", "fev", "mar", "abr", "mai", "jun",
               "jul", "ago", "set", "out", "nov", "dez"]


def _nulo(x) -> bool:
    try:
        return x is None or bool(pd.isna(x))
    except (TypeError, ValueError):
        return False


def _ptbr(texto: str) -> str:
    """Converte '1,234.56' (padrão US) em '1.234,56' (padrão BR)."""
    return texto.replace(",", "§").replace(".", ",").replace("§", ".")


def numero(x, casas: int = 0) -> str:
    return "—" if _nulo(x) else _ptbr(f"{x:,.{casas}f}")


def brl(x, casas: int = 0) -> str:
    return "—" if _nulo(x) else "R$ " + numero(x, casas)


def brl_compacto(x) -> str:
    """R$ 1,5 mi · R$ 540 mil · R$ 50."""
    if _nulo(x):
        return "—"
    a = abs(x)
    s = "-" if x < 0 else ""
    if a >= 1e9:
        return f"{s}R$ {_ptbr(f'{a / 1e9:,.2f}')} bi"
    if a >= 1e6:
        return f"{s}R$ {_ptbr(f'{a / 1e6:,.1f}')} mi"
    if a >= 1e3:
        return f"{s}R$ {_ptbr(f'{a / 1e3:,.0f}')} mil"
    return f"{s}R$ {_ptbr(f'{a:,.0f}')}"


def pct(x, casas: int = 1, sinal: bool = False) -> str:
    if _nulo(x):
        return "—"
    texto = f"{x:+.{casas}f}" if sinal else f"{x:.{casas}f}"
    return _ptbr(texto) + "%"


def mes_ano(ts) -> str:
    ts = pd.Timestamp(ts)
    return f"{MESES_ABREV[ts.month - 1]}/{ts.year}"


def data_br(ts) -> str:
    return pd.Timestamp(ts).strftime("%d/%m/%Y")


# ============================================================================
# 2. LEITURA, VALIDAÇÃO E LIMPEZA
# ============================================================================
COLUNAS_OBRIGATORIAS = ["order_id", "data_pedido", "regiao", "categoria",
                        "canal_venda", "quantidade", "valor_total"]
COLUNAS_TEXTO = ["regiao", "categoria", "canal_venda"]
REGIOES_BR = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]


def ler_csv(fonte) -> pd.DataFrame:
    """Lê o CSV (separador ',' ou ';') e confere as colunas obrigatórias."""
    for sep, decimal in ((",", "."), (";", ",")):
        if hasattr(fonte, "seek"):
            fonte.seek(0)
        try:
            df = pd.read_csv(fonte, sep=sep, decimal=decimal)
        except Exception:  # noqa: BLE001 - tenta o próximo separador
            continue
        df.columns = df.columns.astype(str).str.strip()
        if set(COLUNAS_OBRIGATORIAS) <= set(df.columns):
            return df
    raise ValueError("Não foi possível ler o CSV com as colunas esperadas: "
                     + ", ".join(COLUNAS_OBRIGATORIAS) + ".")


def limpar(bruto: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Aplica as regras de tratamento e devolve (base limpa, relatório)."""
    faltam = [c for c in COLUNAS_OBRIGATORIAS if c not in bruto.columns]
    if faltam:
        raise ValueError(f"Colunas ausentes na base: {', '.join(faltam)}")

    df = bruto[COLUNAS_OBRIGATORIAS].copy()
    rel: dict = {"linhas_lidas": int(len(df))}
    rel["nulos_por_coluna"] = {c: int(n) for c, n in df.isna().sum().items() if n}

    # Tipos
    df["data_pedido"] = pd.to_datetime(df["data_pedido"], errors="coerce", format="ISO8601")
    for c in ("order_id", "quantidade", "valor_total"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Textos
    for c in COLUNAS_TEXTO:
        df[c] = df[c].astype("object").where(df[c].notna(), "Não informado")
        df[c] = df[c].astype(str).str.strip().replace("", "Não informado")

    # Exclusões justificadas
    inval_data = df["data_pedido"].isna()
    inval_id = df["order_id"].isna()
    inval_num = df["quantidade"].isna() | df["valor_total"].isna()
    rel["excluidas_data_invalida"] = int(inval_data.sum())
    rel["excluidas_sem_order_id"] = int(inval_id.sum())
    rel["excluidas_valor_ou_quantidade_invalidos"] = int(inval_num.sum())
    df = df[~(inval_data | inval_id | inval_num)].copy()

    df["data_pedido"] = df["data_pedido"].dt.normalize()
    if (df["order_id"] % 1 == 0).all():
        df["order_id"] = df["order_id"].astype("int64")

    # Duplicidades
    dup_exatas = df.duplicated(keep="first")
    rel["duplicadas_exatas_removidas"] = int(dup_exatas.sum())
    df = df[~dup_exatas].reset_index(drop=True)
    rel["linhas_order_id_repetido"] = int(df["order_id"].duplicated(keep=False).sum())

    # Sinalizações (sem alteração)
    rel["valor_negativo"] = int((df["valor_total"] < 0).sum())
    rel["quantidade_nao_positiva"] = int((df["quantidade"] <= 0).sum())
    rel["regioes_fora_do_padrao"] = sorted(set(df["regiao"]) - set(REGIOES_BR))

    # Granularidade
    rel["linhas_finais"] = int(len(df))
    rel["pedidos_distintos"] = int(df["order_id"].nunique())
    if len(df):
        rel["periodo_ini"] = df["data_pedido"].min()
        rel["periodo_fim"] = df["data_pedido"].max()
    if rel["linhas_order_id_repetido"] == 0:
        rel["granularidade"] = (
            "Cada linha é um pedido (order_id único). `valor_total` é o valor do pedido, "
            "portanto somá-lo não duplica o faturamento; `quantidade` é o nº de itens do pedido."
        )
    else:
        rel["granularidade"] = (
            f"Atenção: {rel['linhas_order_id_repetido']:,} linhas compartilham order_id. "
            "Pedidos são contados por order_id distinto, mas confirme se `valor_total` é o valor "
            "da linha ou do pedido antes de interpretar o faturamento."
        ).replace(",", ".")
    return df, rel


# ============================================================================
# 3. MÉTRICAS, SÉRIES, PROJEÇÃO E BACKTESTING
# ============================================================================
METRICAS = {"faturamento": "Faturamento", "pedidos": "Pedidos", "itens": "Itens vendidos"}
COLS_METRICAS = list(METRICAS)
DIAS_SEMANA = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]
PESOS_WMA = np.array([0.2, 0.3, 0.5])  # do mês mais antigo ao mais recente
_PASSO = {"D": "D", "W": "W-MON", "MS": "MS"}


# ----------------------------------------------------------------------------
# Agregações básicas
# ----------------------------------------------------------------------------
def resumo(df: pd.DataFrame, por) -> pd.DataFrame:
    """Faturamento, pedidos DISTINTOS, itens e ticket médio por uma ou mais colunas."""
    por = [por] if isinstance(por, str) else list(por)
    if df.empty:
        return pd.DataFrame(columns=por + COLS_METRICAS + ["ticket_medio"])
    out = df.groupby(por, as_index=False, observed=True).agg(
        faturamento=("valor_total", "sum"),
        pedidos=("order_id", "nunique"),
        itens=("quantidade", "sum"),
    )
    out["ticket_medio"] = out["faturamento"] / out["pedidos"].where(out["pedidos"] > 0)
    return out


def totais(df: pd.DataFrame) -> dict:
    fat = float(df["valor_total"].sum())
    ped = int(df["order_id"].nunique())
    return {
        "faturamento": fat,
        "pedidos": ped,
        "itens": int(df["quantidade"].sum()),
        "ticket_medio": fat / ped if ped else float("nan"),
    }


def filtrar_periodo(df: pd.DataFrame, inicio, fim) -> pd.DataFrame:
    return df[(df["data_pedido"] >= pd.Timestamp(inicio)) & (df["data_pedido"] <= pd.Timestamp(fim))]


def periodo_anterior(inicio, fim) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Período imediatamente anterior, com o mesmo número de dias."""
    inicio, fim = pd.Timestamp(inicio), pd.Timestamp(fim)
    dias = (fim - inicio).days + 1
    fim_ant = inicio - pd.Timedelta(days=1)
    return fim_ant - pd.Timedelta(days=dias - 1), fim_ant


def variacao_pct(atual: float, anterior: float) -> float | None:
    if anterior is None or pd.isna(anterior) or anterior == 0:
        return None
    return (atual / anterior - 1) * 100


# ----------------------------------------------------------------------------
# Séries temporais
# ----------------------------------------------------------------------------
def _inicio_periodo(datas: pd.Series, freq: str) -> pd.Series:
    if freq == "D":
        return datas
    if freq == "W":  # semana de segunda a domingo, rotulada pela segunda-feira
        return datas.dt.to_period("W-SUN").dt.start_time
    if freq == "MS":
        return datas.dt.to_period("M").dt.start_time
    raise ValueError(f"Granularidade inválida: {freq}")


def serie_temporal(df: pd.DataFrame, freq: str, inicio, fim, por: str | None = None,
                   categorias=None) -> pd.DataFrame:
    """Série completa (períodos sem vendas viram 0) entre `inicio` e `fim`.

    freq: 'D' (dia), 'W' (semana iniciada na segunda) ou 'MS' (mês).
    Com `por`, devolve formato longo com uma linha por (período, grupo).
    """
    d = df.assign(periodo=_inicio_periodo(df["data_pedido"], freq))
    chaves = ["periodo"] + ([por] if por else [])
    r = resumo(d, chaves)

    ini_p = _inicio_periodo(pd.Series([pd.Timestamp(inicio)]), freq).iloc[0]
    fim_p = _inicio_periodo(pd.Series([pd.Timestamp(fim)]), freq).iloc[0]
    calendario = pd.date_range(ini_p, fim_p, freq=_PASSO[freq])

    if por is None:
        r = r.set_index("periodo").reindex(calendario)
        r.index.name = "periodo"
    else:
        grupos = list(categorias) if categorias is not None else sorted(r[por].unique())
        idx = pd.MultiIndex.from_product([calendario, grupos], names=["periodo", por])
        r = r.set_index(["periodo", por]).reindex(idx)
    r[COLS_METRICAS] = r[COLS_METRICAS].fillna(0)
    r["pedidos"] = r["pedidos"].astype(int)
    r["itens"] = r["itens"].astype(int)
    r = r.reset_index()
    r["ticket_medio"] = r["faturamento"] / r["pedidos"].where(r["pedidos"] > 0)
    return r


def media_movel(s: pd.Series, janela: int) -> pd.Series:
    return s.rolling(janela, min_periods=janela).mean()


def aviso_periodos_parciais(inicio, fim, freq: str) -> str | None:
    inicio, fim = pd.Timestamp(inicio), pd.Timestamp(fim)
    if freq == "W" and (inicio.dayofweek != 0 or fim.dayofweek != 6):
        return "A primeira e/ou a última semana do período podem estar incompletas (semanas vão de segunda a domingo)."
    if freq == "MS" and (inicio.day != 1 or fim != fim + pd.offsets.MonthEnd(0)):
        return "O primeiro e/ou o último mês do período estão incompletos; compare com cautela."
    return None


def marcar_meses_completos(mensal: pd.DataFrame, inicio, fim, data_max) -> pd.Series:
    """True para meses totalmente cobertos pelo período filtrado e pelos dados."""
    fim_mes = mensal["periodo"] + pd.offsets.MonthEnd(0)
    limite = min(pd.Timestamp(fim), pd.Timestamp(data_max))
    return (mensal["periodo"] >= pd.Timestamp(inicio)) & (fim_mes <= limite)


# ----------------------------------------------------------------------------
# Sazonalidade
# ----------------------------------------------------------------------------
def por_dia_semana(diaria: pd.DataFrame, col: str = "faturamento") -> pd.DataFrame:
    d = diaria.assign(dia=diaria["periodo"].dt.dayofweek)
    out = d.groupby("dia")[col].agg(media="mean", mediana="median", dias="count").reindex(range(7))
    out.index = DIAS_SEMANA
    return out


def matriz_semana_dia(diaria: pd.DataFrame, col: str = "faturamento") -> pd.DataFrame:
    dow = diaria["periodo"].dt.dayofweek
    d = diaria.assign(semana=diaria["periodo"] - pd.to_timedelta(dow, unit="D"), dia=dow)
    tab = d.pivot_table(index="semana", columns="dia", values=col, aggfunc="sum")
    tab = tab.reindex(columns=range(7))
    tab.columns = DIAS_SEMANA
    return tab


# ----------------------------------------------------------------------------
# Projeção (média móvel ponderada 20/30/50) e backtesting
# ----------------------------------------------------------------------------
def ultimo_mes_completo(data_max) -> pd.Timestamp:
    data_max = pd.Timestamp(data_max).normalize()
    if data_max == data_max + pd.offsets.MonthEnd(0):
        return data_max.to_period("M").start_time
    return (data_max.to_period("M") - 1).start_time


def primeiro_mes_completo(data_min) -> pd.Timestamp:
    data_min = pd.Timestamp(data_min).normalize()
    if data_min.day == 1:
        return data_min
    return (data_min.to_period("M") + 1).start_time


def serie_mensal_historica(df_dim: pd.DataFrame, data_min, data_max) -> pd.DataFrame:
    """Meses COMPLETOS disponíveis (nenhum dado posterior ao último mês completo é usado)."""
    primeiro = primeiro_mes_completo(data_min)
    ultimo = ultimo_mes_completo(data_max)
    if ultimo < primeiro:
        return pd.DataFrame(columns=["periodo"] + COLS_METRICAS + ["ticket_medio"])
    fim = ultimo + pd.offsets.MonthEnd(0)
    base = df_dim[df_dim["data_pedido"] <= fim]
    return serie_temporal(base, "MS", primeiro, fim)


def prever_wma(valores, horizonte: int = 2) -> list[float]:
    """Média móvel ponderada dos 3 últimos meses (20%/30%/50%); recursiva para h>1."""
    hist = list(np.asarray(valores, dtype=float))
    if len(hist) < len(PESOS_WMA):
        raise ValueError("São necessários ao menos 3 meses de histórico.")
    previsoes = []
    for _ in range(horizonte):
        p = float(np.dot(PESOS_WMA, hist[-len(PESOS_WMA):]))
        previsoes.append(p)
        hist.append(p)
    return previsoes


def prever_media(valores, horizonte: int = 2) -> list[float]:
    """Referência simples: média histórica de todos os meses disponíveis."""
    return [float(np.mean(np.asarray(valores, dtype=float)))] * horizonte


def projetar(hist: pd.DataFrame, horizonte: int = 2) -> pd.DataFrame:
    meses = pd.date_range(hist["periodo"].iloc[-1] + pd.offsets.MonthBegin(1),
                          periods=horizonte, freq="MS")
    proj = pd.DataFrame({"periodo": meses})
    for col in COLS_METRICAS:
        proj[col] = prever_wma(hist[col].to_numpy(float), horizonte)
    proj["ticket_medio"] = proj["faturamento"] / proj["pedidos"].where(proj["pedidos"] > 0)
    return proj


def backtest(valores, horizonte: int = 2) -> pd.DataFrame:
    """Backtest temporal com origem móvel: em cada origem só usa dados até ela."""
    v = np.asarray(valores, dtype=float)
    n = len(v)
    linhas = []
    for o in range(len(PESOS_WMA) - 1, n - 1):
        hist = v[: o + 1]
        mmp = prever_wma(hist, horizonte)
        med = prever_media(hist, horizonte)
        for h in range(1, horizonte + 1):
            alvo = o + h
            if alvo < n:
                linhas.append((o, alvo, h, v[alvo], mmp[h - 1], med[h - 1]))
    return pd.DataFrame(linhas, columns=["origem", "alvo", "h", "real", "mmp", "media"])


def erros_backtest(bt: pd.DataFrame, minimo: int = 3) -> pd.DataFrame:
    """MAE e MAPE por horizonte; só informa quando há `minimo` janelas de teste."""
    linhas = []
    for h, g in bt.groupby("h"):
        if len(g) < minimo:
            continue
        positivos = g[g["real"] > 0]
        linha = {"h": int(h), "n": int(len(g))}
        for nome in ("mmp", "media"):
            linha[f"mae_{nome}"] = float((g[nome] - g["real"]).abs().mean())
            linha[f"mape_{nome}"] = (
                float(((positivos[nome] - positivos["real"]).abs() / positivos["real"]).mean() * 100)
                if len(positivos) else float("nan")
            )
        linhas.append(linha)
    return pd.DataFrame(linhas)


def backtest_todas(hist: pd.DataFrame, horizonte: int = 2) -> pd.DataFrame:
    partes = []
    for col in COLS_METRICAS:
        e = erros_backtest(backtest(hist[col].to_numpy(float), horizonte))
        if not e.empty:
            e.insert(0, "metrica", col)
            partes.append(e)
    return pd.concat(partes, ignore_index=True) if partes else pd.DataFrame()


# ============================================================================
# 4. GRÁFICOS (Plotly)
# ============================================================================
PALETA = ["#2563eb", "#f59e0b", "#10b981", "#8b5cf6", "#06b6d4", "#ec4899", "#84cc16", "#ef4444"]
COR_PRINCIPAL = PALETA[0]
COR_PROJECAO = "#ea580c"
COR_REFERENCIA = "#dc2626"
ESCALA = [[0, "#eff6ff"], [0.33, "#bfdbfe"], [0.66, "#60a5fa"], [1, "#2563eb"]]

_ROTULOS = {**METRICAS, "ticket_medio": "Ticket médio"}
_EIXO = {  # prefixo/tickformat do eixo e do hover por métrica
    "faturamento": dict(prefixo="R$ ", fmt=",.0f", hover=",.2f"),
    "ticket_medio": dict(prefixo="R$ ", fmt=",.0f", hover=",.2f"),
    "pedidos": dict(prefixo="", fmt=",.0f", hover=",.0f"),
    "itens": dict(prefixo="", fmt=",.0f", hover=",.0f"),
}


def mapa_cores(itens) -> dict:
    """Cor fixa por item (ordem alfabética do conjunto COMPLETO, para não mudar com filtros)."""
    return {k: PALETA[i % len(PALETA)] for i, k in enumerate(sorted(itens))}


def rotulo(metrica: str) -> str:
    return _ROTULOS.get(metrica, metrica)


def formatar(metrica: str, valor) -> str:
    if metrica == "faturamento":
        return brl_compacto(valor)
    if metrica == "ticket_medio":
        return brl(valor, 2)
    return numero(valor)


def _layout(fig: go.Figure, titulo: str | None = None, altura: int = 380, legenda: bool = True) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        title=dict(text=titulo or "", x=0, xanchor="left", font=dict(size=15)),
        height=altura,
        margin=dict(l=10, r=10, t=55 if titulo else 20, b=10),
        separators=",.",
        showlegend=legenda,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
        font=dict(size=12),
    )
    return fig


def _eixo_y(fig: go.Figure, metrica: str, **kw) -> None:
    e = _EIXO[metrica]
    fig.update_yaxes(tickprefix=e["prefixo"], tickformat=e["fmt"], **kw)


# ----------------------------------------------------------------------------
# Séries temporais
# ----------------------------------------------------------------------------
def _fmt_x(freq: str) -> str:
    return "%m/%Y" if freq == "MS" else "%d/%m/%Y"


def fig_serie(serie: pd.DataFrame, metrica: str, freq: str, titulo: str | None = None,
              medias: tuple[int, ...] = (), altura: int = 380) -> go.Figure:
    """Linha da métrica ao longo do tempo; `medias` adiciona médias móveis (em períodos)."""
    e = _EIXO[metrica]
    suavizada = bool(medias)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=serie["periodo"], y=serie[metrica], mode="lines+markers" if freq == "MS" else "lines",
        name=rotulo(metrica), line=dict(color=COR_PRINCIPAL, width=1.2 if suavizada else 2),
        opacity=0.45 if suavizada else 1,
        hovertemplate=f"{e['prefixo']}%{{y:{e['hover']}}}<extra>{rotulo(metrica)}</extra>",
    ))
    cores = [COR_PROJECAO, "#111827", "#059669"]
    for i, n in enumerate(medias):
        fig.add_trace(go.Scatter(
            x=serie["periodo"], y=media_movel(serie[metrica], n), mode="lines",
            name=f"Média móvel de {n} dias", line=dict(color=cores[i % len(cores)], width=2.5),
            hovertemplate=f"{e['prefixo']}%{{y:{e['hover']}}}<extra>Média móvel {n}d</extra>",
        ))
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(tickformat=_fmt_x(freq), hoverformat=_fmt_x(freq), showgrid=False)
    _eixo_y(fig, metrica, gridcolor="#e5e7eb", rangemode="tozero")
    return _layout(fig, titulo, altura, legenda=suavizada)


def fig_series_grupos(serie: pd.DataFrame, grupo: str, metrica: str, freq: str, cores: dict,
                      destaque: list[str] | None = None, titulo: str | None = None,
                      altura: int = 400) -> go.Figure:
    """Uma linha por grupo; grupos fora de `destaque` ficam esmaecidos."""
    e = _EIXO[metrica]
    fig = go.Figure()
    destaque = None if destaque is None else set(destaque)
    for g, d in serie.groupby(grupo, sort=True):
        ativo = destaque is None or g in destaque
        fig.add_trace(go.Scatter(
            x=d["periodo"], y=d[metrica], mode="lines+markers" if freq == "MS" else "lines", name=str(g),
            line=dict(color=cores.get(g, PALETA[0]), width=2.6 if ativo else 1.2),
            opacity=1 if ativo else 0.2, marker=dict(size=5),
            hovertemplate=f"{e['prefixo']}%{{y:{e['hover']}}}<extra>{g}</extra>",
        ))
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(tickformat=_fmt_x(freq), hoverformat=_fmt_x(freq), showgrid=False)
    _eixo_y(fig, metrica, gridcolor="#e5e7eb", rangemode="tozero")
    return _layout(fig, titulo, altura)


def fig_multi_mensal(mensal: pd.DataFrame, titulo: str | None = None) -> go.Figure:
    """Faturamento, pedidos e itens por mês, em três painéis com eixo X compartilhado."""
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        subplot_titles=[METRICAS[c] for c in ("faturamento", "pedidos", "itens")])
    for i, col in enumerate(("faturamento", "pedidos", "itens"), start=1):
        e = _EIXO[col]
        fig.add_trace(go.Scatter(
            x=mensal["periodo"], y=mensal[col], mode="lines+markers", name=METRICAS[col],
            line=dict(color=PALETA[i - 1], width=2.2),
            hovertemplate=f"%{{x|%m/%Y}}<br>{e['prefixo']}%{{y:{e['hover']}}}<extra></extra>",
        ), row=i, col=1)
        fig.update_yaxes(tickprefix=e["prefixo"], tickformat=e["fmt"], gridcolor="#e5e7eb", row=i, col=1)
    fig.update_xaxes(tickformat="%m/%Y", showgrid=False)
    fig.update_xaxes(dtick="M1", row=3, col=1)
    fig.update_xaxes(tickangle=-45, row=3, col=1)
    return _layout(fig, titulo, 640, legenda=False)


def fig_ano_a_ano(mensal: pd.DataFrame, metrica: str, titulo: str | None = None) -> go.Figure:
    """Mesmo mês em anos diferentes, lado a lado."""
    e = _EIXO[metrica]
    d = mensal.assign(ano=mensal["periodo"].dt.year, mes=mensal["periodo"].dt.month)
    fig = go.Figure()
    for i, (ano, g) in enumerate(d.groupby("ano")):
        fig.add_trace(go.Scatter(
            x=g["mes"], y=g[metrica], mode="lines+markers", name=str(ano),
            line=dict(color=PALETA[i % len(PALETA)], width=2.4),
            hovertemplate=f"{ano}: {e['prefixo']}%{{y:{e['hover']}}}<extra></extra>",
        ))
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(tickmode="array", tickvals=list(range(1, 13)), ticktext=MESES_ABREV, showgrid=False)
    _eixo_y(fig, metrica, gridcolor="#e5e7eb", rangemode="tozero")
    return _layout(fig, titulo, 360)


def fig_variacao_mensal(mensal: pd.DataFrame, metrica: str, titulo: str | None = None) -> go.Figure:
    """Variação % de cada mês completo em relação ao mês anterior."""
    var = mensal[metrica].pct_change() * 100
    var.index = mensal["periodo"]
    var = var.dropna()
    cores = [COR_PRINCIPAL if v >= 0 else COR_PROJECAO for v in var]
    fig = go.Figure(go.Bar(
        x=var.index, y=var.values, marker_color=cores,
        text=[pct(v, 1, sinal=True) for v in var.values], textposition="outside", cliponaxis=False,
        hovertemplate="%{x|%m/%Y}: %{y:+.1f}%<extra></extra>",
    ))
    fig.update_xaxes(tickformat="%m/%Y", dtick="M1", tickangle=-45, showgrid=False)
    fig.update_yaxes(ticksuffix="%", gridcolor="#e5e7eb", zeroline=True, zerolinecolor="#6b7280")
    return _layout(fig, titulo, 360, legenda=False)


# ----------------------------------------------------------------------------
# Rankings, composição e comparações
# ----------------------------------------------------------------------------
def fig_ranking(df: pd.DataFrame, cat: str, col: str, titulo: str | None = None,
                referencia: float | None = None, rotulo_ref: str = "", altura: int | None = None) -> go.Figure:
    """Barras horizontais ordenadas (maior no topo)."""
    d = df.sort_values(col, ascending=True)
    fig = go.Figure(go.Bar(
        x=d[col], y=d[cat], orientation="h", marker_color=COR_PRINCIPAL,
        text=[formatar(col, v) for v in d[col]], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{text}<extra></extra>",
    ))
    if referencia is not None and not pd.isna(referencia):
        fig.add_vline(x=referencia, line_dash="dot", line_color=COR_REFERENCIA)
        fig.add_annotation(x=referencia, y=1.02, yref="paper", text=rotulo_ref, showarrow=False,
                           font=dict(color=COR_REFERENCIA, size=11), xanchor="left")
    maximo = max(d[col].max(), referencia or 0)
    fig.update_xaxes(range=[0, maximo * 1.25 if maximo > 0 else 1], showticklabels=False, showgrid=False)
    fig.update_yaxes(title=None)
    return _layout(fig, titulo, altura or max(240, 60 + 42 * len(d)), legenda=False)


def fig_comparacao_duplo(df: pd.DataFrame, cat: str, col_a: str, col_b: str,
                         titulo: str | None = None) -> go.Figure:
    """Dois painéis (ex.: pedidos | itens) com as mesmas categorias, ordenadas pelo primeiro."""
    d = df.sort_values(col_a, ascending=True)
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.04,
                        subplot_titles=[rotulo(col_a), rotulo(col_b)])
    for j, (col, cor) in enumerate(((col_a, PALETA[0]), (col_b, PALETA[2])), start=1):
        fig.add_trace(go.Bar(
            x=d[col], y=d[cat], orientation="h", marker_color=cor,
            text=[numero(v) for v in d[col]], textposition="outside", cliponaxis=False,
            hovertemplate="%{y}: %{text}<extra></extra>",
        ), row=1, col=j)
        fig.update_xaxes(range=[0, max(d[col].max(), 1) * 1.3], showticklabels=False, showgrid=False,
                         row=1, col=j)
    return _layout(fig, titulo, max(240, 90 + 42 * len(d)), legenda=False)


def fig_donut(df: pd.DataFrame, cat: str, col: str, cores: dict, titulo: str | None = None,
              centro: str = "") -> go.Figure:
    d = df.sort_values(col, ascending=False)
    fig = go.Figure(go.Pie(
        labels=d[cat], values=d[col], hole=0.58, sort=False, direction="clockwise",
        marker=dict(colors=[cores.get(c, PALETA[0]) for c in d[cat]], line=dict(color="white", width=2)),
        textinfo="percent", texttemplate="%{percent:.1%}", insidetextfont=dict(color="white"),
        hovertemplate="%{label}<br>%{value:,.0f} (%{percent:.1%})<extra></extra>",
    ))
    if centro:
        fig.add_annotation(text=centro, x=0.5, y=0.5, showarrow=False, font=dict(size=13))
    return _layout(fig, titulo, 360, legenda=True)


def fig_heatmap(tabela: pd.DataFrame, formatador, titulo: str | None = None, rotulo_barra: str = "",
                mostrar_texto: bool = True, altura: int | None = None, prefixo_barra: str = "",
                sufixo_barra: str = "") -> go.Figure:
    """Mapa de calor (linhas no eixo Y, colunas no eixo X); `formatador` converte valor -> texto."""
    texto = [[formatador(v) for v in linha] for linha in tabela.values]
    fig = go.Figure(go.Heatmap(
        z=tabela.values, x=[str(c) for c in tabela.columns], y=[str(i) for i in tabela.index],
        text=texto, texttemplate="%{text}" if mostrar_texto else None, colorscale=ESCALA,
        xgap=2, ygap=2, hoverongaps=False,
        hovertemplate="%{y} · %{x}<br>%{text}<extra></extra>",
        colorbar=dict(title=rotulo_barra, tickprefix=prefixo_barra, ticksuffix=sufixo_barra, thickness=14),
    ))
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_xaxes(side="bottom", title=None)
    return _layout(fig, titulo, altura or max(260, 80 + 48 * len(tabela)), legenda=False)


def fig_barras_100(percentuais: pd.DataFrame, cores: dict, titulo: str | None = None,
                   titulo_legenda: str = "") -> go.Figure:
    """Barras empilhadas a 100%: índice = barras (eixo X), colunas = fatias."""
    fig = go.Figure()
    for col in percentuais.columns:
        v = percentuais[col]
        fig.add_trace(go.Bar(
            x=[str(i) for i in percentuais.index], y=v, name=str(col), marker_color=cores.get(col, PALETA[0]),
            text=[pct(x, 1) if x >= 4 else "" for x in v], textposition="inside",
            insidetextfont=dict(color="white"),
            hovertemplate=f"{col}: %{{y:.1f}}%<extra></extra>",
        ))
    fig.update_layout(barmode="stack", legend_title_text=titulo_legenda)
    fig.update_yaxes(range=[0, 100], ticksuffix="%", gridcolor="#e5e7eb")
    return _layout(fig, titulo, 420)


# ----------------------------------------------------------------------------
# Sazonalidade
# ----------------------------------------------------------------------------
def fig_media_dia_semana(medias: pd.Series, media_geral: float, metrica: str = "faturamento",
                         titulo: str | None = None) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=medias.index, y=medias.values, marker_color=COR_PRINCIPAL,
        text=[formatar(metrica, v) for v in medias.values], textposition="outside", cliponaxis=False,
        hovertemplate="%{x}: %{text}<extra></extra>",
    ))
    fig.add_hline(y=media_geral, line_dash="dot", line_color=COR_REFERENCIA)
    fig.add_annotation(x=1, xref="paper", y=media_geral, text=f"Média geral: {formatar(metrica, media_geral)}",
                       showarrow=False, xanchor="right", yanchor="bottom",
                       font=dict(color=COR_REFERENCIA, size=11))
    _eixo_y(fig, metrica, gridcolor="#e5e7eb", rangemode="tozero")
    fig.update_xaxes(categoryorder="array", categoryarray=DIAS_SEMANA)
    return _layout(fig, titulo, 360, legenda=False)


def fig_boxplot_dia_semana(diaria: pd.DataFrame, col: str = "faturamento",
                           titulo: str | None = None) -> go.Figure:
    d = diaria.assign(dia=[DIAS_SEMANA[i] for i in diaria["periodo"].dt.dayofweek])
    fig = px.box(d, x="dia", y=col, category_orders={"dia": DIAS_SEMANA},
                 color_discrete_sequence=[COR_PRINCIPAL], points="outliers")
    media = diaria[col].mean()
    fig.add_hline(y=media, line_dash="dot", line_color=COR_REFERENCIA)
    fig.add_annotation(x=1, xref="paper", y=media, text=f"Média geral: {formatar(col, media)}",
                       showarrow=False, xanchor="right", yanchor="bottom",
                       font=dict(color=COR_REFERENCIA, size=11))
    fig.update_xaxes(title=None)
    _eixo_y(fig, col, gridcolor="#e5e7eb", title=None)
    return _layout(fig, titulo, 360, legenda=False)


# ----------------------------------------------------------------------------
# Mapa do Brasil
# ----------------------------------------------------------------------------
def _centros(geojson: dict) -> dict:
    """Centro (caixa envolvente) de cada região, para posicionar os rótulos."""
    centros = {}
    for feat in geojson["features"]:
        lons, lats = [], []

        def _percorre(c):
            if isinstance(c[0], (int, float)):
                lons.append(c[0])
                lats.append(c[1])
            else:
                for x in c:
                    _percorre(x)

        _percorre(feat["geometry"]["coordinates"])
        centros[feat["properties"]["regiao"]] = ((min(lons) + max(lons)) / 2, (min(lats) + max(lats)) / 2)
    return centros


def fig_mapa(geojson: dict, reg: pd.DataFrame, col: str, titulo: str | None = None) -> go.Figure:
    """Macrorregiões coloridas pela métrica `col`; regiões fora do filtro ficam cinza."""
    fig = go.Figure(go.Choropleth(
        geojson=geojson, locations=reg["regiao"], z=reg[col], featureidkey="properties.regiao",
        colorscale=ESCALA, marker_line_color="white", marker_line_width=1,
        colorbar=dict(title=rotulo(col), thickness=14, tickprefix="R$ " if col == "faturamento" else ""),
        customdata=[formatar(col, v) for v in reg[col]],
        hovertemplate="%{location}<br>%{customdata}<extra></extra>",
    ))
    centros = _centros(geojson)
    rotulos = reg.dropna(subset=[col])
    fig.add_trace(go.Scattergeo(
        lon=[centros[r][0] for r in rotulos["regiao"] if r in centros],
        lat=[centros[r][1] for r in rotulos["regiao"] if r in centros],
        text=[f"<b>{r}</b><br>{formatar(col, v)}" for r, v in zip(rotulos["regiao"], rotulos[col]) if r in centros],
        mode="text", textfont=dict(size=11, color="#111827"), hoverinfo="skip", showlegend=False,
    ))
    fig.update_geos(fitbounds="locations", visible=False)
    return _layout(fig, titulo, 520, legenda=False)


# ----------------------------------------------------------------------------
# Projeção
# ----------------------------------------------------------------------------
def fig_projecao(hist: pd.DataFrame, proj: pd.DataFrame, metrica: str, mae: pd.Series | None = None,
                 titulo: str | None = None) -> go.Figure:
    """Histórico (linha sólida) e projeção (tracejada, laranja) no mesmo gráfico."""
    e = _EIXO[metrica]
    fig = go.Figure()
    ult_x, ult_y = hist["periodo"].iloc[-1], hist[metrica].iloc[-1]
    x_proj = [ult_x] + list(proj["periodo"])
    y_proj = [ult_y] + list(proj[metrica])

    fig.add_vrect(x0=ult_x, x1=proj["periodo"].iloc[-1] + pd.Timedelta(days=15),
                  fillcolor="rgba(234,88,12,0.06)", line_width=0, layer="below")
    fig.add_annotation(x=proj["periodo"].iloc[0], y=1.0, yref="paper", text="Projeção (estimativa)",
                       showarrow=False, xanchor="left", yanchor="bottom",
                       font=dict(color=COR_PROJECAO, size=11))
    if mae is not None and len(mae) == len(proj):
        sup = list(proj[metrica] + mae.values)
        inf = list(proj[metrica] - mae.values)
        fig.add_trace(go.Scatter(
            x=list(proj["periodo"]) + list(proj["periodo"])[::-1], y=sup + inf[::-1], fill="toself",
            fillcolor="rgba(234,88,12,0.15)", line=dict(width=0), hoverinfo="skip",
            name="± erro absoluto médio do backtest"))
    fig.add_trace(go.Scatter(
        x=hist["periodo"], y=hist[metrica], mode="lines+markers", name="Histórico (real)",
        line=dict(color=COR_PRINCIPAL, width=2.4),
        hovertemplate=f"%{{x|%m/%Y}}: {e['prefixo']}%{{y:{e['hover']}}}<extra>Real</extra>"))
    fig.add_trace(go.Scatter(
        x=x_proj, y=y_proj, mode="lines+markers", name="Projeção",
        line=dict(color=COR_PROJECAO, width=2.4, dash="dash"), marker=dict(symbol="diamond", size=9),
        hovertemplate=f"%{{x|%m/%Y}}: {e['prefixo']}%{{y:{e['hover']}}}<extra>Projeção</extra>"))
    fig.update_xaxes(tickformat="%m/%Y", dtick="M1", tickangle=-45, showgrid=False)
    _eixo_y(fig, metrica, gridcolor="#e5e7eb", rangemode="tozero")
    return _layout(fig, titulo, 420)


# ============================================================================
# 5. MALHA DAS MACRORREGIÕES
# ============================================================================
CAMINHO = Path(__file__).resolve().parent / "data" / "regioes_brasil.geojson"
URL_ESTADOS = ("https://raw.githubusercontent.com/codeforamerica/click_that_hood/"
               "master/public/data/brazil-states.geojson")

UF_REGIAO = {
    "AC": "Norte", "AP": "Norte", "AM": "Norte", "PA": "Norte", "RO": "Norte", "RR": "Norte", "TO": "Norte",
    "AL": "Nordeste", "BA": "Nordeste", "CE": "Nordeste", "MA": "Nordeste", "PB": "Nordeste",
    "PE": "Nordeste", "PI": "Nordeste", "RN": "Nordeste", "SE": "Nordeste",
    "DF": "Centro-Oeste", "GO": "Centro-Oeste", "MT": "Centro-Oeste", "MS": "Centro-Oeste",
    "ES": "Sudeste", "MG": "Sudeste", "RJ": "Sudeste", "SP": "Sudeste",
    "PR": "Sul", "RS": "Sul", "SC": "Sul",
}


def montar_a_partir_dos_estados(estados: dict) -> dict:
    """Agrupa os polígonos dos estados em um MultiPolygon por região."""
    grupos: dict[str, list] = {}
    for feat in estados["features"]:
        regiao = UF_REGIAO.get(feat["properties"].get("sigla"))
        if regiao is None:
            continue
        geom = feat["geometry"]
        poligonos = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        grupos.setdefault(regiao, []).extend(poligonos)
    if len(grupos) != 5:
        raise ValueError("A malha de estados não trouxe as 27 UFs esperadas (coluna 'sigla').")
    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"regiao": r},
             "geometry": {"type": "MultiPolygon", "coordinates": polys}}
            for r, polys in grupos.items()
        ],
    }


def carregar_geojson_regioes() -> dict | None:
    if CAMINHO.exists():
        return json.loads(CAMINHO.read_text(encoding="utf-8"))
    try:
        resposta = requests.get(URL_ESTADOS, timeout=30)
        resposta.raise_for_status()
        geojson = montar_a_partir_dos_estados(resposta.json())
    except Exception:  # noqa: BLE001 - sem rede/arquivo: o app segue sem o mapa
        return None
    try:
        CAMINHO.parent.mkdir(parents=True, exist_ok=True)
        CAMINHO.write_text(json.dumps(geojson), encoding="utf-8")
    except OSError:
        pass
    return geojson


# ============================================================================
# 6. INTERFACE STREAMLIT
# ============================================================================
RAIZ = Path(__file__).resolve().parent
PASTA_DADOS = RAIZ / "data"
GRANULARIDADES = {"Diária": "D", "Semanal": "W", "Mensal": "MS"}
COLS_METRICA = {"Faturamento": "faturamento", "Pedidos": "pedidos", "Itens vendidos": "itens"}

st.set_page_config(page_title="Dashboard de Vendas", page_icon="📊", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(
    "<style>.block-container{padding-top:2rem;} "
    "[data-testid='stMetricValue']{font-size:1.7rem;}</style>",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Utilitários de interface
# ----------------------------------------------------------------------------
def mostrar(fig, key: str) -> None:
    """Exibe um gráfico Plotly na largura do contêiner (tolerante a versões do Streamlit)."""
    for kw in ({"width": "stretch"}, {"use_container_width": True}, {}):
        try:
            st.plotly_chart(fig, key=key, **kw)
            return
        except Exception:  # noqa: BLE001 - tenta a próxima assinatura
            continue
    st.plotly_chart(fig, key=key)


def tabela(df: pd.DataFrame, **kw) -> None:
    for largura in ({"width": "stretch"}, {"use_container_width": True}, {}):
        try:
            st.dataframe(df, hide_index=True, **largura, **kw)
            return
        except Exception:  # noqa: BLE001 - tenta a próxima assinatura
            continue
    st.dataframe(df)


def aviso_dados_insuficientes(msg: str) -> None:
    st.info(f"ℹ️ {msg}")


@st.cache_data(show_spinner="Carregando e validando os dados…")
def carregar(conteudo: bytes | None, caminho: str | None):
    fonte = io.BytesIO(conteudo) if conteudo is not None else caminho
    return limpar(ler_csv(fonte))


@st.cache_resource(show_spinner=False)
def carregar_mapa():
    return carregar_geojson_regioes()


@dataclass
class Ctx:
    df_all: pd.DataFrame      # base completa e limpa
    df_dim: pd.DataFrame      # filtros de região/categoria/canal (todo o histórico)
    df_sel: pd.DataFrame      # df_dim + filtro de período
    ini: pd.Timestamp
    fim: pd.Timestamp
    data_min: pd.Timestamp
    data_max: pd.Timestamp
    regioes_sel: list
    cores_reg: dict
    cores_cat: dict
    cores_canal: dict
    geojson: dict | None


# ----------------------------------------------------------------------------
# Barra lateral: fonte de dados e filtros globais
# ----------------------------------------------------------------------------
bloco_filtros = st.sidebar.container()
bloco_fonte = st.sidebar.container()

with bloco_fonte:
    with st.expander("📁 Fonte de dados", expanded=False):
        upload = st.file_uploader("Enviar outro CSV", type="csv",
                                  help="Colunas: " + ", ".join(COLUNAS_OBRIGATORIAS))
        arquivos = sorted(PASTA_DADOS.glob("*.csv"))
        if upload is not None:
            conteudo, caminho, nome_fonte = upload.getvalue(), None, upload.name
        elif arquivos:
            conteudo, caminho, nome_fonte = None, str(arquivos[0]), arquivos[0].name
        else:
            conteudo = caminho = nome_fonte = None
        if nome_fonte:
            st.caption(f"Em uso: **{nome_fonte}**")

if conteudo is None and caminho is None:
    st.title("📊 Dashboard Executivo de Vendas")
    st.error("Nenhuma base de dados encontrada. Coloque o CSV de pedidos na pasta `data/` "
             "(ou envie um arquivo em **Fonte de dados**, na barra lateral).")
    st.caption("Colunas esperadas: " + ", ".join(COLUNAS_OBRIGATORIAS))
    st.stop()

try:
    df_all, rel = carregar(conteudo, caminho)
except ValueError as erro:
    st.error(f"Erro ao ler a base: {erro}")
    st.stop()

if df_all.empty:
    st.error("A base não possui registros válidos após a validação.")
    st.stop()

data_min, data_max = df_all["data_pedido"].min(), df_all["data_pedido"].max()
opc_regiao = [r for r in REGIOES_BR if r in set(df_all["regiao"])] + sorted(
    set(df_all["regiao"]) - set(REGIOES_BR))
opc_categoria = sorted(df_all["categoria"].unique())
opc_canal = sorted(df_all["canal_venda"].unique())

with bloco_filtros:
    st.header("Filtros")
    periodo = st.date_input("Período de análise", value=(data_min.date(), data_max.date()),
                            min_value=data_min.date(), max_value=data_max.date(),
                            format="DD/MM/YYYY", key="f_periodo")
    sel_regiao = st.multiselect("Região", opc_regiao, default=opc_regiao, key="f_regiao")
    sel_categoria = st.multiselect("Categoria de produto", opc_categoria, default=opc_categoria,
                                   key="f_categoria")
    sel_canal = st.multiselect("Canal de venda", opc_canal, default=opc_canal, key="f_canal")
    st.caption("Os filtros valem para todas as seções. A projeção (seção 4.3) usa todo o histórico até o "
               "último mês completo: ela respeita região, categoria e canal, mas ignora o período.")

if not isinstance(periodo, (tuple, list)) or len(periodo) != 2:
    st.info("Selecione a data final do período na barra lateral.")
    st.stop()
ini, fim = pd.Timestamp(periodo[0]), pd.Timestamp(periodo[1])

mask = (df_all["regiao"].isin(sel_regiao) & df_all["categoria"].isin(sel_categoria)
        & df_all["canal_venda"].isin(sel_canal))
df_dim = df_all[mask]
df_sel = filtrar_periodo(df_dim, ini, fim)

st.title("📊 Dashboard Executivo de Vendas")
st.caption(f"Base: {data_br(data_min)} a {data_br(data_max)} · "
           f"{numero(rel['pedidos_distintos'])} pedidos · Período analisado: "
           f"{data_br(ini)} a {data_br(fim)}")

if df_sel.empty:
    st.warning("Nenhum pedido encontrado para a combinação de filtros selecionada. "
               "Amplie o período ou inclua mais regiões, categorias ou canais.")
    st.stop()

ctx = Ctx(df_all, df_dim, df_sel, ini, fim, data_min, data_max, sel_regiao,
          mapa_cores(opc_regiao), mapa_cores(opc_categoria), mapa_cores(opc_canal),
          carregar_mapa())


# ----------------------------------------------------------------------------
# 1. Visão Geral
# ----------------------------------------------------------------------------
def aba_visao_geral(ctx: Ctx) -> None:
    tot = totais(ctx.df_sel)
    ini_ant, fim_ant = periodo_anterior(ctx.ini, ctx.fim)
    var = None
    if ini_ant >= ctx.data_min:
        ant = totais(filtrar_periodo(ctx.df_dim, ini_ant, fim_ant))
        var = variacao_pct(tot["faturamento"], ant["faturamento"])

    k = st.columns(5)
    k[0].metric("Faturamento", brl_compacto(tot["faturamento"]),
                delta=pct(var, 1, sinal=True) if var is not None else None,
                help=f"{brl(tot['faturamento'], 2)}. A variação compara com o período anterior de mesma "
                     f"duração ({data_br(ini_ant)} a {data_br(fim_ant)}).")
    k[1].metric("Pedidos (distintos)", numero(tot["pedidos"]))
    k[2].metric("Itens vendidos", numero(tot["itens"]))
    k[3].metric("Ticket médio", brl(tot["ticket_medio"], 2), help="Faturamento ÷ pedidos distintos.")
    k[4].metric("Itens por pedido", numero(tot["itens"] / tot["pedidos"], 2))
    if var is None:
        st.caption("Sem período anterior comparável nos dados para calcular a variação do faturamento.")

    st.subheader("Evolução temporal")
    a, b, d = st.columns([2, 2, 2])
    gran = a.radio("Granularidade", list(GRANULARIDADES), horizontal=True, key="v_gran")
    nome_met = b.selectbox("Métrica", list(COLS_METRICA), key="v_met")
    ma7 = d.checkbox("Média móvel de 7 dias", value=False, key="v_ma", disabled=(gran != "Diária"),
                     help="Disponível na granularidade diária: suaviza as oscilações do dia a dia.")
    freq, col = GRANULARIDADES[gran], COLS_METRICA[nome_met]
    serie = serie_temporal(ctx.df_sel, freq, ctx.ini, ctx.fim)
    mostrar(fig_serie(serie, col, freq, f"{nome_met} — visão {gran.lower()}",
                        medias=(7,) if (ma7 and gran == "Diária") else ()), "v_serie")
    aviso = aviso_periodos_parciais(ctx.ini, ctx.fim, freq)
    if aviso:
        st.caption(f"⚠️ {aviso}")

    st.subheader("Destaques do período")
    diaria = serie if freq == "D" else serie_temporal(ctx.df_sel, "D", ctx.ini, ctx.fim)
    melhor = diaria.loc[diaria["faturamento"].idxmax()]
    dow = por_dia_semana(diaria)
    top = {n: resumo(ctx.df_sel, n).sort_values("faturamento", ascending=False).iloc[0]
           for n in ("regiao", "categoria", "canal_venda")}
    share = {n: top[n]["faturamento"] / tot["faturamento"] * 100 for n in top}
    h1, h2 = st.columns(2)
    h1.markdown(
        f"- **Melhor dia:** {data_br(melhor['periodo'])} ({brl(melhor['faturamento'])})\n"
        f"- **Dia da semana mais forte (média):** {dow['media'].idxmax()} "
        f"({brl(dow['media'].max())} por dia)\n"
        f"- **Faturamento médio diário:** {brl(diaria['faturamento'].mean())}"
    )
    h2.markdown(
        f"- **Região líder:** {top['regiao']['regiao']} ({pct(share['regiao'])} do faturamento)\n"
        f"- **Categoria líder:** {top['categoria']['categoria']} ({pct(share['categoria'])})\n"
        f"- **Canal líder:** {top['canal_venda']['canal_venda']} ({pct(share['canal_venda'])})"
    )


# ----------------------------------------------------------------------------
# 2. Distribuição Geográfica
# ----------------------------------------------------------------------------
def aba_geografica(ctx: Ctx) -> None:
    nome_met = st.radio("Métrica do mapa e da comparação", list(COLS_METRICA), horizontal=True, key="g_met")
    col = COLS_METRICA[nome_met]

    reg = (resumo(ctx.df_sel, "regiao").set_index("regiao")
           .reindex(REGIOES_BR).reset_index())
    reg.loc[~reg["regiao"].isin(ctx.regioes_sel), COLS_METRICAS + ["ticket_medio"]] = np.nan
    sel = reg[reg["regiao"].isin(ctx.regioes_sel)].copy()
    sel[COLS_METRICAS] = sel[COLS_METRICAS].fillna(0)
    total_fat = sel["faturamento"].sum()
    sel["participacao"] = sel["faturamento"] / total_fat * 100 if total_fat else np.nan

    esq, dir_ = st.columns([1.15, 1])
    with esq:
        if ctx.geojson is None:
            st.warning("Malha geográfica indisponível (sem arquivo `data/regioes_brasil.geojson` e sem acesso "
                       "à internet). Gere o arquivo com `scripts/gerar_geojson_regioes.py`.")
        else:
            mostrar(fig_mapa(ctx.geojson, reg, col, f"{nome_met} por macrorregião"), "g_mapa")
    with dir_:
        mostrar(fig_ranking(sel, "regiao", col, f"Ranking de regiões — {nome_met.lower()}", altura=300),
                "g_rank")
        mostrar(fig_donut(sel, "regiao", "faturamento", ctx.cores_reg,
                            "Participação no faturamento", centro="Faturamento<br>total"), "g_donut")

    st.subheader("Pedidos e itens vendidos por região")
    mostrar(fig_comparacao_duplo(sel, "regiao", "pedidos", "itens"), "g_comp")
    tabela_reg = sel.sort_values("faturamento", ascending=False).assign(
        Faturamento=lambda d: d["faturamento"].map(brl),
        Participação=lambda d: d["participacao"].map(pct),
        Pedidos=lambda d: d["pedidos"].map(numero),
        Itens=lambda d: d["itens"].map(numero),
        **{"Ticket médio": lambda d: d["ticket_medio"].map(lambda x: brl(x, 2))},
    )[["regiao", "Faturamento", "Participação", "Pedidos", "Itens", "Ticket médio"]].rename(
        columns={"regiao": "Região"})
    with st.expander("Ver tabela de números por região"):
        tabela(tabela_reg)

    st.subheader("Evolução temporal por região")
    a, b = st.columns([1, 2])
    gran = a.radio("Granularidade", list(GRANULARIDADES), index=2, horizontal=True, key="g_gran")
    destaque = b.multiselect("Regiões em destaque (as demais ficam esmaecidas)", ctx.regioes_sel,
                             default=ctx.regioes_sel, key="g_dest")
    freq = GRANULARIDADES[gran]
    serie = serie_temporal(ctx.df_sel, freq, ctx.ini, ctx.fim, por="regiao", categorias=ctx.regioes_sel)
    mostrar(fig_series_grupos(serie, "regiao", col, freq, ctx.cores_reg, destaque=destaque,
                                titulo=f"{nome_met} por região — visão {gran.lower()}"), "g_evol")
    aviso = aviso_periodos_parciais(ctx.ini, ctx.fim, freq)
    if aviso:
        st.caption(f"⚠️ {aviso}")
    st.caption("Leitura: mostra concentração e diferenças regionais de volume. Decisões de alocação de frota "
               "ou estoque exigem dados operacionais (capacidade, custos, prazos) que não estão nesta base.")


# ----------------------------------------------------------------------------
# 3. Categorias e Canais
# ----------------------------------------------------------------------------
def _pivot(df: pd.DataFrame, linhas: str, colunas: str) -> pd.DataFrame:
    p = df.pivot_table(index=linhas, columns=colunas, values="valor_total", aggfunc="sum", fill_value=0)
    p = p[p.sum().sort_values(ascending=False).index]
    return p.loc[p.sum(axis=1).sort_values(ascending=False).index]


def aba_categorias_canais(ctx: Ctx) -> None:
    cat = resumo(ctx.df_sel, "categoria")
    canal = resumo(ctx.df_sel, "canal_venda")
    tot = totais(ctx.df_sel)

    st.subheader("Contribuição ao faturamento")
    e, d = st.columns(2)
    with e:
        mostrar(fig_ranking(cat, "categoria", "faturamento", "Categorias por faturamento"), "c_rank_cat")
    with d:
        mostrar(fig_ranking(canal, "canal_venda", "faturamento", "Canais de venda por faturamento"),
                "c_rank_canal")

    st.subheader("Categoria × canal")
    e, d = st.columns(2)
    with e:
        modo = st.radio("Valores do mapa de calor", ["Valor absoluto (R$)", "% do total"], horizontal=True,
                        key="c_modo")
        piv = _pivot(ctx.df_sel, "categoria", "canal_venda")
        if modo.startswith("Valor"):
            fig = fig_heatmap(piv, brl_compacto, "Faturamento por categoria e canal")
        else:
            soma = piv.values.sum()
            fig = fig_heatmap(piv / soma * 100 if soma else piv, lambda v: pct(v, 1),
                                "Participação no faturamento total", sufixo_barra="%")
        mostrar(fig, "c_heat_canal")
    with d:
        st.markdown("**Composição das categorias em cada canal**")
        por_canal = piv.div(piv.sum(axis=0).replace(0, np.nan), axis=1).fillna(0) * 100
        mostrar(fig_barras_100(por_canal.T, ctx.cores_cat, titulo=None, titulo_legenda="Categoria"),
                "c_100")

    st.subheader("Ticket médio por categoria")
    mostrar(fig_ranking(cat, "categoria", "ticket_medio", "Ticket médio por categoria",
                          referencia=tot["ticket_medio"], rotulo_ref=f"Geral: {brl(tot['ticket_medio'], 2)}"),
            "c_ticket_cat")

    with st.expander("Pedidos e itens vendidos por categoria e canal"):
        e, d = st.columns(2)
        with e:
            mostrar(fig_comparacao_duplo(cat, "categoria", "pedidos", "itens", "Por categoria"), "c_comp_cat")
        with d:
            mostrar(fig_comparacao_duplo(canal, "canal_venda", "pedidos", "itens", "Por canal"), "c_comp_canal")
        st.caption("O desempenho dos canais é lido por faturamento, pedidos, itens e ticket médio. A base não "
                   "traz cliques, sessões, conversão ou usuários únicos, então não há métrica de engajamento.")
    with st.expander("Categoria × região"):
        piv_r = _pivot(ctx.df_sel, "categoria", "regiao")
        mostrar(fig_heatmap(piv_r, brl_compacto, "Faturamento por categoria e região"), "c_heat_reg")


# ----------------------------------------------------------------------------
# 4. Sazonalidade e Planejamento
# ----------------------------------------------------------------------------
def bloco_sazonalidade(ctx: Ctx) -> None:
    st.markdown("### 4.1 Sazonalidade e comportamento temporal")
    diaria = serie_temporal(ctx.df_sel, "D", ctx.ini, ctx.fim)
    if len(diaria) < 14:
        aviso_dados_insuficientes("Selecione ao menos 14 dias para analisar padrões por dia da semana.")
    else:
        pds = por_dia_semana(diaria)
        e, d = st.columns(2)
        with e:
            mostrar(fig_media_dia_semana(pds["media"], diaria["faturamento"].mean(),
                                           titulo="Faturamento médio por dia da semana"), "s_dow")
        with d:
            mostrar(fig_boxplot_dia_semana(diaria, titulo="Distribuição do faturamento diário"), "s_box")
        with st.expander("Mapa de calor: semana × dia da semana"):
            n_sem = st.selectbox("Semanas exibidas", ["Últimas 12", "Últimas 26", "Últimas 52", "Todas"],
                                 index=1, key="s_semanas")
            mat = matriz_semana_dia(diaria)
            if n_sem != "Todas":
                mat = mat.tail(int(n_sem.split()[-1]))
            mat.index = mat.index.strftime("%d/%m/%Y")
            mostrar(fig_heatmap(mat, brl_compacto, "Faturamento por semana (início na segunda) e dia",
                                  mostrar_texto=len(mat) <= 26, altura=max(320, 28 * len(mat) + 100)),
                    "s_heat_sem")

    mensal = serie_temporal(ctx.df_sel, "MS", ctx.ini, ctx.fim)
    completo = marcar_meses_completos(mensal, ctx.ini, ctx.fim, ctx.data_max)
    mensal_ok = mensal[completo].reset_index(drop=True)
    st.markdown("**Evolução mensal e comparação entre períodos**")
    if len(mensal_ok) < 2:
        aviso_dados_insuficientes("O período selecionado tem menos de 2 meses completos; amplie o período "
                                  "para ver a evolução mensal e comparações.")
        return
    if len(mensal_ok) < len(mensal):
        st.caption("Meses parciais (início/fim do período) foram omitidos para não distorcer as comparações.")
    mostrar(fig_multi_mensal(mensal_ok), "s_multi")

    nome_met = st.selectbox("Métrica da comparação entre períodos", list(COLS_METRICA), key="s_met_comp")
    col = COLS_METRICA[nome_met]
    e, d = st.columns(2)
    with e:
        mostrar(fig_ano_a_ano(mensal_ok, col, f"{nome_met}: mesmo mês em anos diferentes"), "s_yoy")
        if mensal_ok["periodo"].dt.year.nunique() < 2:
            st.caption("Há apenas um ano nos dados selecionados: não há comparação ano a ano.")
    with d:
        mostrar(fig_variacao_mensal(mensal_ok, col, f"{nome_met}: variação sobre o mês anterior"), "s_mom")


def bloco_ticket_demanda(ctx: Ctx) -> None:
    st.markdown("### 4.2 Ticket médio e indicadores de demanda")
    st.info("A base **não contém dados de estoque**. Por isso não são calculados giro, cobertura, ruptura ou "
            "estoque de segurança. A quantidade de itens vendidos é usada apenas como aproximação da demanda "
            "observada.")
    mensal = serie_temporal(ctx.df_sel, "MS", ctx.ini, ctx.fim)
    completo = marcar_meses_completos(mensal, ctx.ini, ctx.fim, ctx.data_max)
    mensal_ok = mensal[completo].reset_index(drop=True)
    tot = totais(ctx.df_sel)

    e, d = st.columns(2)
    with e:
        if len(mensal_ok) < 2:
            aviso_dados_insuficientes("São necessários ao menos 2 meses completos para a evolução do ticket médio.")
        else:
            fig = fig_serie(mensal_ok, "ticket_medio", "MS", "Evolução mensal do ticket médio", altura=360)
            fig.add_hline(y=tot["ticket_medio"], line_dash="dot", line_color=COR_REFERENCIA)
            mostrar(fig, "t_ticket")
    with d:
        tk = ctx.df_sel.pivot_table(index="categoria", columns="regiao", values="valor_total", aggfunc="sum")
        pe = ctx.df_sel.pivot_table(index="categoria", columns="regiao", values="order_id", aggfunc="nunique")
        tk = (tk / pe.where(pe > 0))
        tk = tk.loc[tk.mean(axis=1).sort_values(ascending=False).index]
        mostrar(fig_heatmap(tk, lambda v: brl(v, 0), "Ticket médio: categoria × região",
                              altura=360, prefixo_barra="R$ "), "t_heat")

    st.markdown("**Evolução diária de itens vendidos (demanda observada)**")
    diaria = serie_temporal(ctx.df_sel, "D", ctx.ini, ctx.fim)
    if len(diaria) < 30:
        aviso_dados_insuficientes("Selecione ao menos 30 dias para exibir as médias móveis de 7 e 30 dias.")
    else:
        mostrar(fig_serie(diaria, "itens", "D", medias=(7, 30), altura=360), "t_itens")

    st.markdown("**Indicadores de demanda por categoria e região**")
    dias = (ctx.fim - ctx.ini).days + 1
    total_itens = tot["itens"]

    def demanda(por: str, nome: str) -> pd.DataFrame:
        r = resumo(ctx.df_sel, por).sort_values("itens", ascending=False)
        return pd.DataFrame({
            nome: r[por], "Itens vendidos": r["itens"].map(numero),
            "% dos itens": (r["itens"] / total_itens * 100).map(pct),
            "Itens por dia": (r["itens"] / dias).map(lambda x: numero(x, 1)),
            "Pedidos por dia": (r["pedidos"] / dias).map(lambda x: numero(x, 1)),
            "Itens por pedido": (r["itens"] / r["pedidos"]).map(lambda x: numero(x, 2)),
        })

    e, d = st.columns(2)
    with e:
        tabela(demanda("categoria", "Categoria"))
    with d:
        tabela(demanda("regiao", "Região"))


def bloco_projecao(ctx: Ctx) -> None:
    st.markdown("### 4.3 Projeção para os próximos dois meses")
    st.caption("Estimativas **indicativas para planejamento**, não previsões garantidas. Usam todo o histórico "
               "até o último mês completo (o filtro de período não se aplica) e respeitam região, categoria e canal.")
    hist = serie_mensal_historica(ctx.df_dim, ctx.data_min, ctx.data_max)
    if len(hist) < 3:
        aviso_dados_insuficientes("São necessários ao menos 3 meses completos de histórico para projetar.")
        return
    proj = projetar(hist)
    bt = backtest_todas(hist)
    ultimo = hist.iloc[-1]

    linhas = [{"Mês": f"{mes_ano(ultimo['periodo'])} (real)", "Faturamento": brl(ultimo["faturamento"]),
               "Pedidos": numero(ultimo["pedidos"]), "Itens vendidos": numero(ultimo["itens"]),
               "Ticket médio": brl(ultimo["ticket_medio"], 2)}]
    for _, r in proj.iterrows():
        linhas.append({"Mês": f"{mes_ano(r['periodo'])} (projeção)", "Faturamento": brl(r["faturamento"]),
                       "Pedidos": numero(r["pedidos"]), "Itens vendidos": numero(r["itens"]),
                       "Ticket médio": brl(r["ticket_medio"], 2)})
    tabela(pd.DataFrame(linhas))

    nome_met = st.selectbox("Métrica exibida no gráfico", list(COLS_METRICA), key="p_met")
    col = COLS_METRICA[nome_met]
    mae = None
    if not bt.empty:
        sub = bt[bt["metrica"] == col].set_index("h")
        if {1, 2} <= set(sub.index):
            mae = pd.Series([sub.loc[1, "mae_mmp"], sub.loc[2, "mae_mmp"]])
    mostrar(fig_projecao(hist, proj, col, mae, f"{nome_met}: histórico mensal e projeção"), "p_graf")

    st.markdown("**Backtesting temporal: média móvel ponderada × média histórica**")
    if bt.empty:
        aviso_dados_insuficientes("Histórico insuficiente para backtesting (mínimo de 6 meses completos).")
    else:
        t = bt.copy()
        t["Métrica"] = t["metrica"].map(METRICAS)
        t["Horizonte"] = t["h"].map({1: "1 mês à frente", 2: "2 meses à frente"})
        t["Melhor (MAPE)"] = np.where(t["mape_mmp"] <= t["mape_media"], "Média móvel ponderada",
                                      "Média histórica")
        fmt_erro = lambda r, k: (brl(r[k]) if r["metrica"] == "faturamento" else numero(r[k]))  # noqa: E731
        exibe = pd.DataFrame({
            "Métrica": t["Métrica"], "Horizonte": t["Horizonte"], "Janelas testadas": t["n"],
            "MAPE – média móvel pond.": t["mape_mmp"].map(pct),
            "MAPE – média histórica": t["mape_media"].map(pct),
            "MAE – média móvel pond.": t.apply(lambda r: fmt_erro(r, "mae_mmp"), axis=1),
            "MAE – média histórica": t.apply(lambda r: fmt_erro(r, "mae_media"), axis=1),
            "Melhor (MAPE)": t["Melhor (MAPE)"],
        })
        tabela(exibe)
        venceu = int((t["mape_mmp"] <= t["mape_media"]).sum())
        st.caption(f"A média móvel ponderada teve menor MAPE que a média histórica em {venceu} de {len(t)} "
                   "comparações. Em cada janela, só dados anteriores à origem foram usados (sem vazamento).")

    with st.expander("Metodologia e limitações"):
        st.markdown(
            """
**Método.** Média móvel ponderada dos **3 últimos meses completos**, com pesos de **20%, 30% e 50%**
(do mês mais antigo ao mais recente), aplicada a faturamento, pedidos distintos e itens vendidos.
O 2º mês projetado usa o 1º mês projetado como se fosse observado (projeção recursiva). O ticket médio
projetado é a razão entre o faturamento e os pedidos projetados.

**Sem vazamento.** Só entram meses completos; o mês corrente incompleto é ignorado. No backtesting, cada
previsão usa apenas dados até a sua origem.

**Backtesting.** Origem móvel: para cada mês a partir do 3º, projeta 1 e 2 meses à frente e compara com o
realizado. A referência é a **média histórica** (média de todos os meses até a origem). Métricas: **MAPE**
(erro percentual médio) e **MAE** (erro absoluto médio). A faixa sombreada do gráfico é ± o MAE do
backtesting para cada horizonte, uma ordem de grandeza do erro, não um intervalo de confiança.

**Limitações.**
- Com cerca de 18 meses não é possível estimar sazonalidade anual; o método extrapola o nível recente.
- Meses têm quantidades diferentes de dias e dias úteis, o que não é ajustado.
- Não considera promoções, mudanças de preço, eventos ou capacidade operacional.
- Em filtros muito específicos (poucos pedidos) a série fica instável e o erro tende a ser alto.
            """
        )


def aba_sazonalidade_planejamento(ctx: Ctx) -> None:
    bloco_sazonalidade(ctx)
    st.divider()
    bloco_ticket_demanda(ctx)
    st.divider()
    bloco_projecao(ctx)


# ----------------------------------------------------------------------------
# Estrutura principal: exatamente quatro abas
# ----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Visão Geral", "Distribuição Geográfica", "Categorias e Canais",
                                  "Sazonalidade e Planejamento"])
with tab1:
    aba_visao_geral(ctx)
with tab2:
    aba_geografica(ctx)
with tab3:
    aba_categorias_canais(ctx)
with tab4:
    aba_sazonalidade_planejamento(ctx)

with st.sidebar.expander("🧹 Qualidade e tratamento dos dados"):
    st.markdown(f"**Granularidade:** {rel['granularidade']}")
    st.markdown(
        f"- Linhas lidas: {numero(rel['linhas_lidas'])} · linhas finais: {numero(rel['linhas_finais'])}\n"
        f"- Excluídas por data inválida: {rel['excluidas_data_invalida']}\n"
        f"- Excluídas sem `order_id`: {rel['excluidas_sem_order_id']}\n"
        f"- Excluídas por valor/quantidade inválidos: {rel['excluidas_valor_ou_quantidade_invalidos']}\n"
        f"- Duplicadas exatas removidas: {rel['duplicadas_exatas_removidas']}\n"
        f"- Linhas com `order_id` repetido: {rel['linhas_order_id_repetido']}\n"
        f"- Valores negativos (mantidos): {rel['valor_negativo']}\n"
        f"- Quantidade ≤ 0 (mantidas): {rel['quantidade_nao_positiva']}"
    )
    if rel["nulos_por_coluna"]:
        st.markdown(f"Nulos por coluna: {rel['nulos_por_coluna']}")
    if rel["regioes_fora_do_padrao"]:
        st.warning(f"Regiões fora das 5 macrorregiões: {', '.join(rel['regioes_fora_do_padrao'])}")
    st.caption("Registros só são excluídos quando não podem ser posicionados no tempo, contados ou somados; "
               "inconsistências possíveis são apenas sinalizadas.")
