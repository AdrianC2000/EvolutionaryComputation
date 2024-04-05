import tkinter as tk
from tkinter import ttk, messagebox

from genetic_algorithm import GeneticAlgorithm


class GeneticAlgorithmGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Genetic Algorithm Parameters")

        # Default values
        self.variables_number = tk.IntVar(value=3)
        self.bounds_min = tk.IntVar(value=-10)
        self.bounds_max = tk.IntVar(value=10)
        self.population_size = tk.IntVar(value=100)
        self.epochs_number = tk.IntVar(value=100)
        self.precision = tk.IntVar(value=6)
        self.selection_methods = tk.StringVar(value="tournament")
        self.tournaments_count = tk.IntVar(value=3)
        self.fraction_selected = tk.DoubleVar(value=0.34)
        self.crossover_method = tk.StringVar(value="single-point")
        self.crossover_probability = tk.DoubleVar(value=0.7)
        self.mutation_method = tk.StringVar(value="single-point")
        self.mutation_rate = tk.DoubleVar(value=0.1)
        self.inversion_probability = tk.DoubleVar(value=0.5)
        self.elitism_ratio = tk.DoubleVar(value=0.5)
        self.min_max = tk.StringVar(value="max")

        # Frame for variables
        variables_frame = ttk.LabelFrame(root, text="Variables")
        variables_frame.grid(row=0, column=0, padx=10, pady=5, sticky="w")

        # Variables input
        ttk.Label(variables_frame, text="Number of variables:").grid(row=0, column=0, sticky="w")
        ttk.Entry(variables_frame, textvariable=self.variables_number, width=10).grid(row=0, column=1, sticky="w")

        ttk.Label(variables_frame, text="Bounds:").grid(row=1, column=0, sticky="w")
        ttk.Entry(variables_frame, textvariable=self.bounds_min, width=5).grid(row=1, column=1, sticky="w")
        ttk.Entry(variables_frame, textvariable=self.bounds_max, width=5).grid(row=1, column=2, sticky="w")

        ttk.Label(variables_frame, text="Population size:").grid(row=2, column=0, sticky="w")
        ttk.Entry(variables_frame, textvariable=self.population_size, width=10).grid(row=2, column=1, sticky="w")

        ttk.Label(variables_frame, text="Epochs number:").grid(row=3, column=0, sticky="w")
        ttk.Entry(variables_frame, textvariable=self.epochs_number, width=10).grid(row=3, column=1, sticky="w")

        ttk.Label(variables_frame, text="Precision:").grid(row=4, column=0, sticky="w")
        ttk.Entry(variables_frame, textvariable=self.precision, width=10).grid(row=4, column=1, sticky="w")

        # Frame for selection methods
        selection_frame = ttk.LabelFrame(root, text="Selection")
        selection_frame.grid(row=1, column=0, padx=10, pady=5, sticky="w")

        ttk.Label(selection_frame, text="Selection method:").grid(row=0, column=0, sticky="w")
        ttk.Combobox(selection_frame, textvariable=self.selection_methods,
                     values=["tournament", "best", "roulette"]).grid(row=0, column=1, sticky="w")

        ttk.Label(selection_frame, text="Tournaments count:").grid(row=1, column=0, sticky="w")
        ttk.Entry(selection_frame, textvariable=self.tournaments_count, width=10).grid(row=1, column=1, sticky="w")

        ttk.Label(selection_frame, text="Fraction selected:").grid(row=2, column=0, sticky="w")
        ttk.Entry(selection_frame, textvariable=self.fraction_selected, width=10).grid(row=2, column=1, sticky="w")

        # Frame for crossover and mutation
        crossover_mutation_frame = ttk.LabelFrame(root, text="Crossover and Mutation")
        crossover_mutation_frame.grid(row=2, column=0, padx=10, pady=5, sticky="w")

        ttk.Label(crossover_mutation_frame, text="Crossover method:").grid(row=0, column=0, sticky="w")
        ttk.Combobox(crossover_mutation_frame, textvariable=self.crossover_method,
                     values=["single-point", "mssx"]).grid(row=0, column=1, sticky="w")

        ttk.Label(crossover_mutation_frame, text="Crossover probability:").grid(row=1, column=0, sticky="w")
        ttk.Entry(crossover_mutation_frame, textvariable=self.crossover_probability, width=10).grid(row=1, column=1,
                                                                                                    sticky="w")

        ttk.Label(crossover_mutation_frame, text="Mutation method:").grid(row=2, column=0, sticky="w")
        ttk.Combobox(crossover_mutation_frame, textvariable=self.mutation_method,
                     values=["single-point", "other"]).grid(row=2, column=1, sticky="w")

        ttk.Label(crossover_mutation_frame, text="Mutation rate:").grid(row=3, column=0, sticky="w")
        ttk.Entry(crossover_mutation_frame, textvariable=self.mutation_rate, width=10).grid(row=3, column=1, sticky="w")

        ttk.Label(crossover_mutation_frame, text="Inversion probability:").grid(row=4, column=0, sticky="w")
        ttk.Entry(crossover_mutation_frame, textvariable=self.inversion_probability, width=10).grid(row=4, column=1,
                                                                                                    sticky="w")

        ttk.Label(crossover_mutation_frame, text="Elitism ratio:").grid(row=5, column=0, sticky="w")
        ttk.Entry(crossover_mutation_frame, textvariable=self.elitism_ratio, width=10).grid(row=5, column=1,
                                                                                            sticky="w")

        # Frame for min/max
        min_max_frame = ttk.LabelFrame(root, text="Min/Max")
        min_max_frame.grid(row=3, column=0, padx=10, pady=5, sticky="w")

        ttk.Radiobutton(min_max_frame, text="Min", variable=self.min_max, value="min").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(min_max_frame, text="Max", variable=self.min_max, value="max").grid(row=0, column=1, sticky="w")

        # Run button
        ttk.Button(root, text="Run", command=self.run_algorithm).grid(row=4, column=0, padx=10, pady=5)

    def run_algorithm(self):
        bounds = (self.bounds_min.get(), self.bounds_max.get())
        is_min_searched = self.min_max.get() == "min"

        genetic_algorithm = GeneticAlgorithm(self.precision.get(), bounds, self.variables_number.get(),
                                             self.selection_methods.get(), self.crossover_method.get(),
                                             self.crossover_probability.get(), self.mutation_method.get(),
                                             self.mutation_rate.get(), self.elitism_ratio.get(), is_min_searched,
                                             self.tournaments_count.get(), self.fraction_selected.get())

        best_individual, best_fitness = genetic_algorithm.find_best_solution(self.population_size.get(),
                                                                             self.epochs_number.get())

        output_text = f"Best found individual: {best_individual}\nFitness: {best_fitness}"
        messagebox.showinfo("Algorithm Output", output_text)


if __name__ == "__main__":
    root = tk.Tk()
    app = GeneticAlgorithmGUI(root)
    root.mainloop()
