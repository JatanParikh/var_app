import streamlit as st
import pandas as pd
import numpy as np
import datetime
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.stats as ststats
import math

st.set_page_config(page_title="Portfolio VaR Tool", layout="wide")
st.title("📊 Portfolio VaR & ES Tool")

# -----------------------------
# Sidebar inputs
# -----------------------------
st.sidebar.header("Portfolio Inputs")

n_stocks = st.sidebar.number_input(
    "Number of stocks in portfolio", min_value=1, max_value=30, value=4, step=1
)

raw_stocks = []
raw_weights = []
for i in range(n_stocks):
    t = st.sidebar.text_input(f"Ticker Stock {i+1} (e.g., RELIANCE.NS)")
    w = st.sidebar.number_input(f"Weight Stock {i+1}", min_value=0.0, max_value=1.0, step=0.01, value=0.0)
    raw_stocks.append((t or "").strip().upper())
    raw_weights.append(w)

start_date = st.sidebar.date_input("Start Date", datetime.date(2020, 1, 1))
benchmark = st.sidebar.text_input("Benchmark Ticker", value="^NSEI").strip().upper()

risk_free_rate_annual = st.sidebar.number_input(
    "Risk-free rate (annual, decimal)", min_value=0.0, max_value=0.2, value=0.0225, step=0.0025
)

# VaR/ES controls
alpha = st.sidebar.slider("Tail probability α (VaR/ES level)", min_value=0.01, max_value=0.10, value=0.05, step=0.01)
h_days = st.sidebar.number_input("Holding period (days)", min_value=1, max_value=60, value=1, step=1)

export_csv = st.sidebar.checkbox("Export price CSV")

# Filter UI entries
pairs = [(s, w) for s, w in zip(raw_stocks, raw_weights) if s and w > 0]

if st.sidebar.button("Retrieve Data"):
    try:
        if not pairs:
            st.error("Please enter at least one valid stock ticker with weight > 0.")
            st.stop()

        req_tickers = [s for s, _ in pairs]
        req_weights = np.array([w for _, w in pairs], dtype=float)

        # -----------------------------
        # Download stock prices
        # -----------------------------
        raw = yf.download(req_tickers, start=start_date, progress=False)
        if "Adj Close" in raw.columns:
            px = raw["Adj Close"]
        elif "Close" in raw.columns:
            px = raw["Close"]
        else:
            st.error("No valid price column ('Adj Close' or 'Close') found for stocks.")
            st.stop()

        # Ensure DataFrame even for single ticker
        if isinstance(px, pd.Series):
            px = px.to_frame()

        # Drop entirely empty columns (bad tickers)
        px = px.dropna(axis=1, how="all")

        if px.shape[1] == 0:
            st.error("No valid stock price data returned. Check ticker symbols (use .NS/.BO for India).")
            st.stop()

        returned_tickers = list(px.columns)
        # Align weights to what actually returned
        w_map = dict(pairs)
        aligned_weights = np.array([w_map[t] for t in returned_tickers if t in w_map], dtype=float)

        # Warn if some requested tickers didn't come back
        missing = [t for t in req_tickers if t not in returned_tickers]
        if missing:
            st.warning(f"Dropped missing/invalid tickers: {', '.join(missing)}")

        if aligned_weights.size == 0 or aligned_weights.sum() == 0:
            st.error("All requested tickers were missing or had zero weights after alignment.")
            st.stop()

        # Renormalize weights after dropping missing tickers
        if abs(aligned_weights.sum() - 1.0) > 1e-8:
            aligned_weights = aligned_weights / aligned_weights.sum()
            st.info("Weights renormalized to sum to 1.0 after dropping missing tickers.")

        # -----------------------------
        # Download benchmark
        # -----------------------------
        raw_bench = yf.download(benchmark, start=start_date, progress=False)
        if "Adj Close" in raw_bench.columns:
            bench_px = raw_bench["Adj Close"]
        elif "Close" in raw_bench.columns:
            bench_px = raw_bench["Close"]
        else:
            st.error("No valid price column in benchmark data.")
            st.stop()
        # If single-col DF, squeeze
        if isinstance(bench_px, pd.DataFrame) and bench_px.shape[1] == 1:
            bench_px = bench_px.squeeze()
        bench_px = bench_px.dropna()

        # -----------------------------
        # Returns
        # -----------------------------
        stock_rets = px.pct_change()

        # Date-wise weight renormalization to handle missing data
        w_series = pd.Series(aligned_weights, index=returned_tickers)
        weighted = stock_rets.mul(w_series, axis=1)
        eff_w = (~stock_rets.isna()).mul(w_series, axis=1).sum(axis=1)
        portfolio_ret = (weighted.sum(axis=1) / eff_w).dropna()

        # Horizon aggregation: convert daily returns to h-day compounded returns
        if h_days == 1:
            port_h = portfolio_ret.copy()
        else:
            port_h = (1 + portfolio_ret).rolling(h_days).apply(lambda x: np.prod(1.0 + x) - 1.0, raw=True).dropna()

        bench_ret = bench_px.pct_change().dropna()
        if h_days == 1:
            bench_h = bench_ret.copy()
        else:
            bench_h = (1 + bench_ret).rolling(h_days).apply(lambda x: np.prod(1.0 + x) - 1.0, raw=True).dropna()

        # Cumulative for chart (daily)
        portfolio_cum = (1 + portfolio_ret).cumprod()
        bench_cum = (1 + bench_ret).cumprod()

        # -----------------------------
        # Risk / Performance metrics
        # -----------------------------
        mean_d = portfolio_ret.mean()
        std_d = portfolio_ret.std()
        ann_ret = mean_d * 252
        ann_std = std_d * np.sqrt(252)
        sharpe = (ann_ret - risk_free_rate_annual) / ann_std if ann_std != 0 else np.nan

        aligned = pd.concat([portfolio_ret, bench_ret], axis=1, join="inner").dropna()
        aligned.columns = ["port", "bench"]

        TE = (aligned["port"] - aligned["bench"]).std() if len(aligned) else np.nan
        beta = (
            np.cov(aligned["port"], aligned["bench"])[0, 1] / aligned["bench"].var()
            if len(aligned) and aligned["bench"].var() != 0
            else np.nan
        )

        # -----------------------------
        # VaR & ES (at horizon h_days, tail α)
        # -----------------------------
        def hist_var_es(x: pd.Series, alpha: float):
            if len(x) == 0:
                return np.nan, np.nan
            var = x.quantile(alpha)
            es = x[x <= var].mean() if (x <= var).any() else np.nan
            return var, es

        # Historical
        h_var, h_es = hist_var_es(port_h.dropna(), alpha)

        # Parametric (Normal) on horizon distribution (use port_h mean/std directly)
        mu_h = port_h.mean()
        sigma_h = port_h.std()
        if sigma_h and not np.isnan(sigma_h):
            z = ststats.norm.ppf(alpha)
            p_var = mu_h + sigma_h * z
            # ES for Normal: μ - σ * φ(z) / α  (left tail)
            p_es = mu_h - sigma_h * (ststats.norm.pdf(z) / alpha)
        else:
            p_var, p_es = np.nan, np.nan

        # Monte Carlo: simulate daily, compound to horizon
        # Simulate from daily normal (μ, σ) estimated from daily portfolio_ret
        mu_d = portfolio_ret.mean()
        sigma_d = portfolio_ret.std()
        if sigma_d and not np.isnan(sigma_d):
            sims = 20000
            rng = np.random.default_rng(42)
            if h_days == 1:
                sim_h = rng.normal(mu_d, sigma_d, size=sims)
            else:
                # simulate h-days of daily returns and compound
                sim_h = []
                for _ in range(sims):
                    dr = rng.normal(mu_d, sigma_d, size=h_days)
                    sim_h.append(np.prod(1.0 + dr) - 1.0)
                sim_h = np.array(sim_h)
            mc_var = np.quantile(sim_h, alpha)
            mc_es = sim_h[sim_h <= mc_var].mean() if (sim_h <= mc_var).any() else np.nan
        else:
            mc_var, mc_es = np.nan, np.nan

        # -----------------------------
        # UI tabs
        # -----------------------------
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📈 Portfolio Data", "📊 Descriptive Data", "⚠️ VaR & ES", "🔗 Correlation Matrix"]
        )

        with tab1:
            st.subheader("Portfolio & Benchmark Performance (Cumulative, Daily)")
            perf_df = pd.concat(
                [portfolio_cum.rename("Portfolio"), bench_cum.rename("Benchmark")],
                axis=1,
            ).dropna()
            st.line_chart(perf_df)

            st.write("### Headline Metrics")
            st.metric("Mean Daily Return", f"{mean_d*100:.2f}%")
            st.metric("Daily Std Dev", f"{std_d*100:.2f}%")
            st.metric("Annualized Return", f"{ann_ret*100:.2f}%")
            st.metric("Annualized Std Dev", f"{ann_std*100:.2f}%")
            st.metric("Sharpe Ratio (annual)", f"{sharpe:.2f}")
            st.metric("Tracking Error (daily)", f"{TE*100:.2f}%" if pd.notna(TE) else "—")
            st.metric("Beta vs Benchmark", f"{beta:.2f}" if pd.notna(beta) else "—")

            if missing:
                st.caption("Note: Some tickers were dropped due to missing data.")

        with tab2:
            st.subheader(f"Historic Return Distributions (daily)")
            fig, ax = plt.subplots(figsize=(6,4))
            for c in px.columns:
                ax.hist(stock_rets[c].dropna(), bins=50, alpha=0.5, label=c, density=True)
            ax.hist(portfolio_ret, bins=50, alpha=0.5, label="Portfolio", density=True)
            ax.legend()
            st.pyplot(fig)

        with tab3:
            st.subheader(f"VaR & Expected Shortfall at α={alpha:.2f}, horizon={h_days} day(s)")

            # --- Numbers table
            var_es_table = pd.DataFrame({
                "Method": ["Historical", "Parametric (Normal)", "Monte Carlo"],
                "VaR (return)": [h_var, p_var, mc_var],
                "ES / CVaR (return)": [h_es, p_es, mc_es],
            })
            st.dataframe(var_es_table.style.format({"VaR (return)": "{:.2%}", "ES / CVaR (return)": "{:.2%}"}), use_container_width=True)

            # --- Plot A: Histogram of horizon returns (Historical) with VaR/ES lines
            if len(port_h) >= 10:
                fig1, ax1 = plt.subplots(figsize=(6,4))
                ax1.hist(port_h, bins=60, density=True, alpha=0.7)
                if pd.notna(h_var):
                    ax1.axvline(h_var, linestyle="--", linewidth=2, label=f"Hist VaR ({alpha:.0%})")
                if pd.notna(h_es):
                    ax1.axvline(h_es, linestyle=":", linewidth=2, label=f"Hist ES ({alpha:.0%})")
                ax1.set_title("Historical horizon returns")
                ax1.legend()
                st.pyplot(fig1)

            # --- Plot B: Parametric normal curve with VaR/ES lines
            if pd.notna(mu_h) and pd.notna(sigma_h) and sigma_h > 0:
                xs = np.linspace(mu_h - 5*sigma_h, mu_h + 5*sigma_h, 800)
                ys = ststats.norm.pdf(xs, loc=mu_h, scale=sigma_h)
                fig2, ax2 = plt.subplots(figsize=(6,4))
                ax2.plot(xs, ys)
                if pd.notna(p_var):
                    ax2.axvline(p_var, linestyle="--", linewidth=2, label=f"Param VaR ({alpha:.0%})")
                if pd.notna(p_es):
                    ax2.axvline(p_es, linestyle=":", linewidth=2, label=f"Param ES ({alpha:.0%})")
                ax2.set_title("Parametric (Normal) density of horizon returns")
                ax2.legend()
                st.pyplot(fig2)

            # --- Plot C: Monte Carlo histogram with VaR/ES lines
            if pd.notna(mc_var):
                fig3, ax3 = plt.subplots(figsize=(6,4))
                # Recreate MC sample for plotting if not available
                mu_d = portfolio_ret.mean()
                sigma_d = portfolio_ret.std()
                rng = np.random.default_rng(7)
                if h_days == 1:
                    sim_h_plot = rng.normal(mu_d, sigma_d, size=40000)
                else:
                    sim_h_plot = []
                    for _ in range(40000):
                        dr = rng.normal(mu_d, sigma_d, size=h_days)
                        sim_h_plot.append(np.prod(1.0 + dr) - 1.0)
                    sim_h_plot = np.array(sim_h_plot)
                ax3.hist(sim_h_plot, bins=80, density=True, alpha=0.7)
                ax3.axvline(mc_var, linestyle="--", linewidth=2, label=f"MC VaR ({alpha:.0%})")
                if pd.notna(mc_es):
                    ax3.axvline(mc_es, linestyle=":", linewidth=2, label=f"MC ES ({alpha:.0%})")
                ax3.set_title("Monte Carlo horizon returns")
                ax3.legend()
                st.pyplot(fig3)

        with tab4:
            st.subheader("Correlation Matrix (daily returns, overlapping dates)")
            corr = stock_rets.loc[portfolio_ret.index].corr()
            st.dataframe(corr, use_container_width=True)

            fig, ax = plt.subplots(figsize=(6,4))
            cax = ax.matshow(corr, cmap="Blues")
            plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
            plt.yticks(range(len(corr.columns)), corr.columns)
            fig.colorbar(cax)
            st.pyplot(fig)

        if export_csv:
            px.to_csv("portfolio_prices.csv")
            st.success("✅ Exported prices to portfolio_prices.csv")

    except Exception as e:
        st.error(f"Error loading data: {e}")
