import numpy as np
import pandas as pd

class FTSLeeManual:
    def __init__(self, data):
        self.data_series = data['Kurs Jual'].values
        self.prediksi = None
        self.aktual = None
        self.mape = None
        self.interval = None
        self.z1 = None
        self.z2 = None

        self.run()

    def run(self):
        data_series = self.data_series
        N = len(data_series)
        K = 1 + 3.322 * np.log10(N)
        jumlah_himpunan = round(K)

        Dmin = np.min(data_series)
        Dmax = np.max(data_series)

        Z1 = 50
        Z2 = 50
        self.z1 = Z1
        self.z2 = Z2

        U_start = Dmin - Z1
        U_end = Dmax + Z2
        R = U_end - U_start
        l = R / jumlah_himpunan

        intervals = [U_start + i * l for i in range(jumlah_himpunan + 1)]
        intervals = np.array(intervals)

        predictions = self.run_fts_lee(data_series, intervals)
        actual = data_series[:len(predictions)]
        interval_tuples = [(round(intervals[i], 2), round(intervals[i + 1], 2)) for i in range(len(intervals) - 1)]

        self.prediksi = predictions
        self.aktual = actual
        self.mape = self.calculate_mape(actual[1:], predictions[1:])  # Lewati prediksi pertama (NaN)
        self.interval = interval_tuples

    def calculate_mape(self, actual, predicted):
        actual, predicted = np.array(actual), np.array(predicted)
        return np.mean(np.abs((actual - predicted) / actual)) * 100

    def fuzzyfikasi(self, kurs, df_himpunan):
        for _, row in df_himpunan.iterrows():
            if row['Batas Bawah'] <= kurs <= row['Batas Atas']:
                return row['Kelas Interval(Ai)']
        return "Unknown"

    def calculate_flr(self, df_data):
        flr = ["NaN"]
        for i in range(1, len(df_data)):
            current_state = df_data.iloc[i]['Fuzzyfikasi']
            next_state = df_data.iloc[i + 1]['Fuzzyfikasi'] if i + 1 < len(df_data) else current_state
            flr.append(f"{current_state} → {next_state}")
        df_data['FLR'] = flr

    def calculate_flrg(self, df_data):
        flrg = {}
        for i in range(1, len(df_data)):
            current_flr = df_data.iloc[i]['FLR']
            if isinstance(current_flr, str) and '→' in current_flr:
                from_state, to_state = current_flr.split(' → ')
                if from_state not in flrg:
                    flrg[from_state] = []
                flrg[from_state].append(to_state)
        return flrg

    def get_fuzzy_representations(self, min_val, max_val, jumlah_himpunan):
        interval = (max_val - min_val) / jumlah_himpunan
        representasi = {}
        for i in range(jumlah_himpunan):
            lower = min_val + i * interval
            upper = lower + interval
            mean = (lower + upper) / 2
            representasi[f"A{i+1}"] = round(mean, 2)
        return representasi

    def defuzzify_lee(self, df_data, flrg, representasi):
        prediksi_list = []
        for i in range(len(df_data)):
            current_state = df_data.iloc[i]['Fuzzyfikasi']
            if current_state in flrg:
                next_states = flrg[current_state]
                nilai_anggota = [representasi[state] for state in next_states if state in representasi]
                total = sum(nilai_anggota)
                count = len(nilai_anggota)
                prediksi = round(total / count, 2) if count > 0 else float('nan')
            else:
                prediksi = float('nan')
            prediksi_list.append(prediksi)
        df_data['Prediksi'] = prediksi_list
        return df_data

    def run_fts_lee(self, data_series, intervals):
        jumlah_himpunan = len(intervals) - 1
        df_himpunan = pd.DataFrame({
            "Kelas Interval(Ai)": [f"A{i+1}" for i in range(jumlah_himpunan)],
            "Batas Bawah": intervals[:-1],
            "Batas Atas": intervals[1:]
        })

        df_data = pd.DataFrame(data_series, columns=["Kurs Jual"])
        df_data['No'] = df_data.index + 1
        df_data['Fuzzyfikasi'] = df_data['Kurs Jual'].apply(lambda x: self.fuzzyfikasi(x, df_himpunan))

        self.calculate_flr(df_data)
        flrg = self.calculate_flrg(df_data)
        representasi = self.get_fuzzy_representations(np.min(data_series), np.max(data_series), jumlah_himpunan)
        df_data = self.defuzzify_lee(df_data, flrg, representasi)

        predictions = df_data['Prediksi'].values  # Ambil semua, termasuk NaN di awal
        return predictions
