import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model_pso import PSOOptimizer
from fts_manual import FTSLeeManual

st.set_page_config(
    page_title="Prediksi Nilai Tukar Rupiah",
    page_icon="📈",
    layout="wide",
)

class PSOFTSApp:
    def __init__(self):
        if "page" not in st.session_state:
            st.session_state.page = "input"
        self.route_page()

    def route_page(self):
        if st.session_state.page == "input":
            self.page_input()
        elif st.session_state.page == "output":
            self.page_output()

    def page_input(self):
        st.markdown("<h1 style='text-align: center;'>📈 Prediksi Nilai Tukar Rupiah</h1>", unsafe_allow_html=True)

        uploaded_file = st.file_uploader("📥 Upload file data kurs:", type=None)

        if uploaded_file is not None:
            filename = uploaded_file.name
            if not filename.endswith(".csv"):
                st.error("❌ Format file salah. Harap unggah file .csv.")
                return

            try:
                data = pd.read_csv(uploaded_file)
                data['Tanggal'] = pd.to_datetime(data['Tanggal'])
                data = data.sort_values(by='Tanggal', ascending=True).reset_index(drop=True)

                st.session_state.uploaded_file = uploaded_file
                st.session_state.data = data
                st.success("✅ Data berhasil dimuat!")

            except Exception as e:
                st.error("❌ Terjadi kesalahan saat membaca file:")
                st.exception(e)
                return

        elif "data" in st.session_state:
            data = st.session_state.data
        else:
            return

        st.write("📄 **Preview Data (5 teratas):**", st.session_state.data.head())

        kurs_jual = st.session_state.data['Kurs Jual'].values
        min_val, max_val = kurs_jual.min(), kurs_jual.max()
        jumlah_data = len(kurs_jual)

        st.markdown(f"""
        <div style='background-color:#e6f2ff;padding:20px;border-radius:10px;margin-top:20px;'>
            <h4>📊 Informasi Data</h4>
            <ul style='list-style:none;padding-left:0;font-size:16px;'>
                <li>🔹 <strong>Nilai Minimum:</strong> {min_val:.2f}</li>
                <li>🔸 <strong>Nilai Maksimum:</strong> {max_val:.2f}</li>
                <li>📈 <strong>Jumlah Data:</strong> {jumlah_data}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("⚙️ Pengaturan Parameter PSO"):
            st.session_state.n_particles = st.number_input("🔹 Jumlah Partikel", value=st.session_state.get("n_particles", 10), min_value=5, max_value=500)
            st.session_state.n_iterations = st.number_input("🔸 Jumlah Iterasi", value=st.session_state.get("n_iterations", 30), min_value=10, max_value=500)
            st.session_state.c1 = st.number_input("🔹c1 (koefisien kognitif)", value=st.session_state.get("c1", 1.5))
            st.session_state.c2 = st.number_input("🔸c2 (koefisien sosial)", value=st.session_state.get("c2", 1.5))
            st.session_state.w = st.number_input("🔹w (bobot inersia)", value=st.session_state.get("w", 0.9))

        if st.button("Jalankan Prediksi"):
            with st.spinner("⏳ Sedang melakukan perhitungan..."):
                fts_manual = FTSLeeManual(st.session_state.data)

                st.session_state.prediksi_manual = np.insert(fts_manual.prediksi[1:], 0, np.nan)
                st.session_state.aktual_manual = fts_manual.aktual
                st.session_state.mape_manual = fts_manual.mape
                st.session_state.interval_manual = fts_manual.interval
                st.session_state.z1_manual = fts_manual.z1
                st.session_state.z2_manual = fts_manual.z2
                st.session_state.jumlah_interval_manual = len(fts_manual.interval)

                optimizer = PSOOptimizer(
                    st.session_state.data,
                    st.session_state.n_particles,
                    st.session_state.n_iterations,
                    w=st.session_state.w,
                    c1=st.session_state.c1,
                    c2=st.session_state.c2
                )

                st.session_state.prediksi = np.insert(optimizer.prediksi[1:], 0, np.nan)
                st.session_state.aktual = optimizer.aktual
                st.session_state.interval = optimizer.get_interval_tuples()
                st.session_state.mape = optimizer.gbest_score
                st.session_state.z1 = optimizer.z1_best
                st.session_state.z2 = optimizer.z2_best
                result_df = optimizer.get_result_dataframe()
                result_df["Prediksi"] = st.session_state.prediksi  # Update prediksi dengan NaN awal
                st.session_state.result_df = result_df
                st.session_state.jumlah_interval = len(st.session_state.interval)

                st.session_state.page = "output"
                st.rerun()

    def page_output(self):
        st.markdown("<h1 style='text-align: center;'>📈 Prediksi Nilai Tukar Rupiah", unsafe_allow_html=True)

        st.markdown(f"""
        <div style='background-color:#f5f5f5;padding:15px;border-radius:10px;margin-bottom:20px;'>
            <h4>⚙️ Parameter PSO yang Digunakan:</h4>
            <ul style='list-style:none;padding-left:0;font-size:16px;'>
                <li>🔹 Jumlah Partikel: <strong>{st.session_state.n_particles}</strong></li>
                <li>🔸 Jumlah Iterasi: <strong>{st.session_state.n_iterations}</strong></li>
                <li>🔹c1: <strong>{st.session_state.c1}</strong></li>
                <li>🔸c2: <strong>{st.session_state.c2}</strong></li>
                <li>🔹w: <strong>{st.session_state.w}</strong></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🔹 FTS Lee")
            st.markdown(f"- Z1: `{st.session_state.z1_manual:.2f}`")
            st.markdown(f"- Z2: `{st.session_state.z2_manual:.2f}`")
            st.markdown(f"- Jumlah Interval: `{st.session_state.jumlah_interval_manual}`")
            st.markdown("**📌 Interval FTS Lee:**")
            st.dataframe(pd.DataFrame(st.session_state.interval_manual, columns=["Batas Bawah", "Batas Atas"]))

        with col2:
            st.markdown("### 🔸 FTS Lee + PSO")
            st.markdown(f"- Z1 Terbaik: `{st.session_state.z1:.2f}`")
            st.markdown(f"- Z2 Terbaik: `{st.session_state.z2:.2f}`")
            st.markdown(f"- Jumlah Interval Terbaik: `{st.session_state.jumlah_interval}`")
            st.markdown("**📌 Interval FTS Lee + PSO:**")
            st.dataframe(pd.DataFrame(st.session_state.interval, columns=["Batas Bawah", "Batas Atas"]))

        st.markdown("---")
        st.markdown("### 📉 Plot dan Tabel Hasil Prediksi")

        col1, col2 = st.columns(2)

        df_manual = pd.DataFrame({
            "No": range(1, len(st.session_state.aktual_manual) + 1),
            "Tanggal": st.session_state.data['Tanggal'].dt.strftime('%d-%m-%Y'),
            "Aktual": st.session_state.aktual_manual,
            "Prediksi": st.session_state.prediksi_manual
        })

        with col1:
            st.subheader("🔹 FTS Lee")
            st.dataframe(df_manual)
            st.markdown(f"<h5 style='color: green;'>MAPE: {st.session_state.mape_manual:.4f}%</h5>", unsafe_allow_html=True)
            fig_manual, ax_manual = plt.subplots(figsize=(8, 4))
            ax_manual.plot(st.session_state.data['Tanggal'], st.session_state.aktual_manual, label='Aktual', color='blue', marker='o')
            ax_manual.plot(st.session_state.data['Tanggal'], st.session_state.prediksi_manual, label='Prediksi', color='red', linestyle='--', marker='x')
            ax_manual.set_title("FTS Lee: Aktual vs Prediksi")
            ax_manual.set_xlabel("Tanggal")
            ax_manual.set_ylabel("Kurs Jual")
            ax_manual.legend()
            fig_manual.autofmt_xdate(rotation=45)
            st.pyplot(fig_manual)

        df_pso = pd.DataFrame({
            "No": range(1, len(st.session_state.aktual) + 1),
            "Tanggal": st.session_state.data['Tanggal'].dt.strftime('%d-%m-%Y'),
            "Aktual": st.session_state.aktual,
            "Prediksi": st.session_state.prediksi
        })

        with col2:
            st.subheader("🔸 FTS Lee + PSO")
            st.dataframe(df_pso)
            st.markdown(f"<h5 style='color: green;'>MAPE: {st.session_state.mape:.4f}%</h5>", unsafe_allow_html=True)
            fig_pso, ax_pso = plt.subplots(figsize=(8, 4))
            ax_pso.plot(st.session_state.data['Tanggal'], st.session_state.aktual, label='Aktual', color='blue', marker='o')
            ax_pso.plot(st.session_state.data['Tanggal'], st.session_state.prediksi, label='Prediksi', color='red', linestyle='--', marker='x')
            ax_pso.set_title("FTS Lee + PSO: Aktual vs Prediksi")
            ax_pso.set_xlabel("Tanggal")
            ax_pso.set_ylabel("Kurs Jual")
            ax_pso.legend()
            fig_pso.autofmt_xdate(rotation=45)
            st.pyplot(fig_pso)

        st.markdown("### 🔀 Perbandingan Prediksi FTS Lee VS FTS Lee PSO", unsafe_allow_html=True)

        fig_all, ax_all = plt.subplots(figsize=(12, 5))
        ax_all.plot(st.session_state.data['Tanggal'], st.session_state.aktual, label='Aktual', color='green', linewidth=2)
        ax_all.plot(st.session_state.data['Tanggal'], st.session_state.prediksi_manual, label='Prediksi FTS Lee', color='navy', linewidth=2)
        ax_all.plot(st.session_state.data['Tanggal'], st.session_state.prediksi, label='Prediksi FTS Lee + PSO', color='red', linewidth=2)
        ax_all.set_title("Hasil Prediksi", fontsize=14, weight='bold')
        ax_all.set_xlabel("Tanggal", fontsize=12)
        ax_all.set_ylabel("Nilai Kurs Jual", fontsize=12)
        ax_all.legend()
        ax_all.grid(True)
        fig_all.autofmt_xdate(rotation=45)

        with st.container():
            cols = st.columns([1, 6, 1])
            with cols[1]:
                st.pyplot(fig_all)

        st.markdown("---")
        if st.button("🔄 Kembali ke Input"):
            st.session_state.page = "input"
            st.rerun()

# Jalankan Aplikasi
if __name__ == "__main__":
    PSOFTSApp()
