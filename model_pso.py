import numpy as np
import pandas as pd

class PSOOptimizer:
    def __init__(self, data, n_particles, n_iterations, w, c1, c2):
        self.data_series = np.asarray(data['Kurs Jual'].values, dtype=float)
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.Dmin = np.min(self.data_series)
        self.Dmax = np.max(self.data_series)
        self.range_data = self.Dmax - self.Dmin
        self.set_z_ranges()

        self.n_intervals = np.random.randint(5, 16)
        self.gbest = None
        self.gbest_score = np.inf
        self.mape_per_iter = []
        self.prediksi = None
        self.aktual = None
        self.best_intervals = None
        self.z1_best = None
        self.z2_best = None

        self.run()

    def set_z_ranges(self):
        if self.range_data <= 100:
            self.z1_range = self.z2_range = (0, 30)
        elif self.range_data <= 300:
            self.z1_range = self.z2_range = (10, 50)
        else:
            self.z1_range = self.z2_range = (30, 100)

    def calculate_mape(self, actual, predicted):
        return np.mean(np.abs((actual - predicted) / actual)) * 100

    def generate_intervals(self, z1, z2, interval_points):
        return np.concatenate(([self.Dmin - z1], interval_points, [self.Dmax + z2]))

    def get_fuzzy_classes(self, intervals):
        return (intervals[:-1] + intervals[1:]) / 2

    def fuzzify_series(self, series, intervals):
        fuzzy = np.zeros_like(series, dtype=int)
        for i in range(len(intervals) - 1):
            fuzzy[(series >= intervals[i]) & (series <= intervals[i + 1])] = i
        return fuzzy

    def build_flrg(self, fuzzified):
        flrg = {i: [] for i in range(np.max(fuzzified) + 1)}
        for i in range(len(fuzzified) - 1):
            flrg[fuzzified[i]].append(fuzzified[i + 1])
        return flrg

    def defuzzify(self, fuzzified, flrg, reps):
        pred = []
        for f in fuzzified:
            next_states = flrg.get(f, [])
            if next_states:
                pred.append(np.mean([reps[s] for s in next_states]))
            else:
                pred.append(np.nan)
        return np.array(pred)

    def run_fts_lee(self, intervals):
        fuzzy_classes = self.get_fuzzy_classes(intervals)
        fuzzified = self.fuzzify_series(self.data_series, intervals)
        flrg = self.build_flrg(fuzzified)
        pred = self.defuzzify(fuzzified, flrg, fuzzy_classes)
        return pred  # kembalikan semua prediksi, termasuk NaN

    def initialize_particles(self):
        particles = []
        for _ in range(self.n_particles):
            z1 = np.random.uniform(*self.z1_range)
            z2 = np.random.uniform(*self.z2_range)
            interval_points = np.sort(np.random.uniform(self.Dmin, self.Dmax, self.n_intervals - 1))
            particle = np.concatenate(([z1, z2], interval_points))
            particles.append(particle)
        return np.array(particles)

    def run(self):
        particles = self.initialize_particles()
        velocities = np.zeros_like(particles)
        pbest = particles.copy()
        pbest_scores = np.full(self.n_particles, np.inf)

        for iter_num in range(self.n_iterations):
            for i in range(self.n_particles):
                z1, z2 = particles[i, 0], particles[i, 1]
                interval_points = particles[i, 2:]
                full_intervals = self.generate_intervals(z1, z2, interval_points)
                predictions = self.run_fts_lee(full_intervals)

                # Lewati nilai pertama karena prediksi pertama adalah NaN
                actual = self.data_series[1:len(predictions)]
                mape = self.calculate_mape(actual, predictions[1:])

                if mape < pbest_scores[i]:
                    pbest_scores[i] = mape
                    pbest[i] = particles[i].copy()
                if mape < self.gbest_score:
                    self.gbest_score = mape
                    self.gbest = particles[i].copy()

            self.mape_per_iter.append(self.gbest_score)

            print(f"[Iter {iter_num + 1:03d}] MAPE = {self.gbest_score:.4f}% | z1 = {self.gbest[0]:.2f}, z2 = {self.gbest[1]:.2f}, intervals = {len(self.gbest[2:]) + 1}")

            r1, r2 = np.random.rand(self.n_particles, particles.shape[1]), np.random.rand(self.n_particles, particles.shape[1])
            velocities = (
                self.w * velocities
                + self.c1 * r1 * (pbest - particles)
                + self.c2 * r2 * (self.gbest - particles)
            )
            particles += velocities
            particles[:, 0:2] = np.clip(particles[:, 0:2], self.z1_range[0], self.z1_range[1])
            particles[:, 2:] = np.clip(particles[:, 2:], self.Dmin, self.Dmax)
            for i in range(self.n_particles):
                particles[i, 2:] = np.sort(particles[i, 2:])

        self.z1_best, self.z2_best = self.gbest[0], self.gbest[1]
        self.best_intervals = self.generate_intervals(self.z1_best, self.z2_best, self.gbest[2:])
        self.prediksi = self.run_fts_lee(self.best_intervals)
        self.aktual = self.data_series[:len(self.prediksi)]  # tetap simpan panjang yang sama

    def get_result_dataframe(self):
        return pd.DataFrame({
            "No": range(1, len(self.prediksi) + 1),
            "Aktual": self.aktual,
            "Prediksi": self.prediksi
        })

    def get_interval_tuples(self):
        return [(round(self.best_intervals[i], 2), round(self.best_intervals[i + 1], 2))
                for i in range(len(self.best_intervals) - 1)]
