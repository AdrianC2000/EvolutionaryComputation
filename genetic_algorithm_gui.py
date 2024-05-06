import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from genetic_algorithms.genetic_algorithm_real import GeneticAlgorithmReal
from genetic_algorithms.genetic_algorithms_binary import GeneticAlgorithmBinary


class GeneticAlgorithmGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Genetic Algorithm Parameters")

        # Default values
        self.binary_real = tk.StringVar(value="binary")
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
        self.fitness_function = tk.StringVar(value="square_sum")

        # Tabs for binary/real
        tab_control = ttk.Notebook(root)
        tab_real = ttk.Frame(tab_control)
        tab_control.add(tab_real, text='Real')

        tab_binary = ttk.Frame(tab_control)
        tab_control.add(tab_binary, text='Binary')

        # Frame for fitness function
        fitness_frame_binary = ttk.LabelFrame(tab_binary, text="Fitness")
        fitness_frame_real = ttk.LabelFrame(tab_real, text="Fitness")
        self.generate_fitness_frame(fitness_frame_binary)
        self.generate_fitness_frame(fitness_frame_real)

        # Frame for variables
        variables_frame_binary = ttk.LabelFrame(tab_binary, text="Variables")
        variables_frame_real = ttk.LabelFrame(tab_real, text="Variables")

        # Variables input
        self.generate_variables_input(variables_frame_binary)
        self.generate_variables_input(variables_frame_real)

        # Frame for selection methods
        selection_frame_binary = ttk.LabelFrame(tab_binary, text="Selection")
        selection_frame_real = ttk.LabelFrame(tab_real, text="Selection")
        self.generate_selection_frame(selection_frame_binary)
        self.generate_selection_frame(selection_frame_real)

        # Frame for crossover and mutation
        crossover_mutation_frame_binary = ttk.LabelFrame(tab_binary, text="Crossover and Mutation")
        crossover_mutation_frame_real = ttk.LabelFrame(tab_real, text="Crossover and Mutation")
        self.generate_crossover_mutation_frame(crossover_mutation_frame_binary, is_real=False)
        self.generate_crossover_mutation_frame(crossover_mutation_frame_real, is_real=True)

        # Frame for min/max
        min_max_frame_binary = ttk.LabelFrame(tab_binary, text="Min/Max")
        min_max_frame_real = ttk.LabelFrame(tab_real, text="Min/Max")
        self.generate_min_max_frame(min_max_frame_binary)
        self.generate_min_max_frame(min_max_frame_real)

        # Run button
        ttk.Button(tab_binary, text="Run", command=self.run_algorithm).grid(row=5, column=0, padx=10, pady=5)
        ttk.Button(tab_real, text="Run", command=self.run_algorithm).grid(row=5, column=0, padx=10, pady=5)
        tab_control.pack(expand=1, fill="both")

    def generate_min_max_frame(self, min_max_frame):
        min_max_frame.grid(row=4, column=0, padx=10, pady=5, sticky="w")
        ttk.Radiobutton(min_max_frame, text="Min", variable=self.min_max, value="min").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(min_max_frame, text="Max", variable=self.min_max, value="max").grid(row=0, column=1, sticky="w")

    def generate_crossover_mutation_frame(self, crossover_mutation_frame, is_real: bool):
        if is_real:
            crossover_methods = ["arithmetical", "linear", "blend_alpha", "blend_alpha_and_beta", "average",
                                 "differential_evolution", "unfair_average", "gaussian_uniform"]
            mutation_methods = ["uniform", "gaussian"]
        else:
            crossover_methods = ["single-point", "two-point", "three-point", "uniform", "grain", "mssx", "three-parent",
                                 "nonuniform"]
            mutation_methods = ["single-point"]

        crossover_mutation_frame.grid(row=3, column=0, padx=10, pady=5, sticky="w")
        ttk.Label(crossover_mutation_frame, text="Crossover method:").grid(row=0, column=0, sticky="w")

        ttk.Combobox(
            crossover_mutation_frame,
            textvariable=self.crossover_method,
            values=crossover_methods
        ).grid(row=0, column=1, sticky="w")

        ttk.Label(crossover_mutation_frame, text="Crossover probability:").grid(row=1, column=0, sticky="w")
        ttk.Entry(crossover_mutation_frame, textvariable=self.crossover_probability, width=10).grid(row=1, column=1,
                                                                                                    sticky="w")
        ttk.Label(crossover_mutation_frame, text="Mutation method:").grid(row=2, column=0, sticky="w")
        ttk.Combobox(crossover_mutation_frame, textvariable=self.mutation_method,
                     values=mutation_methods).grid(row=2, column=1, sticky="w")
        ttk.Label(crossover_mutation_frame, text="Mutation rate:").grid(row=3, column=0, sticky="w")
        ttk.Entry(crossover_mutation_frame, textvariable=self.mutation_rate, width=10).grid(row=3, column=1, sticky="w")
        ttk.Label(crossover_mutation_frame, text="Inversion probability:").grid(row=4, column=0, sticky="w")
        ttk.Entry(crossover_mutation_frame, textvariable=self.inversion_probability, width=10).grid(row=4, column=1,
                                                                                                    sticky="w")
        ttk.Label(crossover_mutation_frame, text="Elitism ratio:").grid(row=5, column=0, sticky="w")
        ttk.Entry(crossover_mutation_frame, textvariable=self.elitism_ratio, width=10).grid(row=5, column=1,
                                                                                            sticky="w")

    def generate_selection_frame(self, selection_frame):
        selection_frame.grid(row=2, column=0, padx=10, pady=5, sticky="w")
        ttk.Label(selection_frame, text="Selection method:").grid(row=0, column=0, sticky="w")
        ttk.Combobox(selection_frame, textvariable=self.selection_methods,
                     values=["tournament", "best", "roulette"]).grid(row=0, column=1, sticky="w")
        ttk.Label(selection_frame, text="Tournaments count:").grid(row=1, column=0, sticky="w")
        ttk.Entry(selection_frame, textvariable=self.tournaments_count, width=10).grid(row=1, column=1, sticky="w")
        ttk.Label(selection_frame, text="Fraction selected:").grid(row=2, column=0, sticky="w")
        ttk.Entry(selection_frame, textvariable=self.fraction_selected, width=10).grid(row=2, column=1, sticky="w")

    def generate_fitness_frame(self, fitness_frame):
        fitness_frame.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        ttk.Label(fitness_frame, text="Fitness function:").grid(row=0, column=0, sticky="w")
        ttk.Combobox(fitness_frame, textvariable=self.fitness_function,
                     values=["square_sum", "katsuura", "rana"]).grid(row=0, column=1, sticky="w")

    def generate_variables_input(self, variables_frame):
        variables_frame.grid(row=1, column=0, padx=10, pady=5, sticky="w")
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

    def plot_results(self, genetic_algorithm):
        # Create a new Tkinter window
        plot_window = tk.Toplevel(self.root)
        plot_window.title("GA Performance Metrics")

        # Create the figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plotting the value of the function from each iteration
        ax1.plot(genetic_algorithm.fitness_history, label='Best Value')
        ax1.set_title('Best Value Over Iterations')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Value')
        ax1.legend()

        # Plotting the average and standard deviation
        ax2.errorbar(range(len(genetic_algorithm.average_fitness_history)), genetic_algorithm.average_fitness_history,
                     yerr=genetic_algorithm.std_dev_fitness_history, label='Average Value', fmt='-o')
        ax2.set_title('Average Value and Standard Deviation Over Iterations')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Value')
        ax2.legend()

        # Adding the plot to the Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def run_algorithm(self):
        bounds = (self.bounds_min.get(), self.bounds_max.get())
        is_min_searched = self.min_max.get() == "min"

        if self.binary_real.get() == "binary":
            genetic_algorithm = GeneticAlgorithmBinary(self.precision.get(), bounds, self.variables_number.get(),
                                                       self.selection_methods.get(), self.crossover_method.get(),
                                                       self.crossover_probability.get(), self.mutation_method.get(),
                                                       self.mutation_rate.get(), self.inversion_probability.get(),
                                                       self.elitism_ratio.get(), self.fitness_function.get(),
                                                       is_min_searched, self.tournaments_count.get(),
                                                       self.fraction_selected.get())
        else:
            genetic_algorithm = GeneticAlgorithmReal(bounds, self.variables_number.get(), self.selection_methods.get(),
                                                     self.crossover_method.get(), self.crossover_probability.get(),
                                                     self.mutation_method.get(), self.mutation_rate.get(),
                                                     self.elitism_ratio.get(), self.fitness_function.get(),
                                                     is_min_searched, self.tournaments_count.get(),
                                                     self.fraction_selected.get())

        best_individual, best_fitness = genetic_algorithm.find_best_solution(self.population_size.get(),
                                                                             self.epochs_number.get())

        output_text = f"Best found individual: {best_individual}\nFitness: {best_fitness}"
        self.plot_results(genetic_algorithm)
        messagebox.showinfo("Algorithm Output", output_text)


if __name__ == "__main__":
    root = tk.Tk()
    app = GeneticAlgorithmGUI(root)
    root.mainloop()
