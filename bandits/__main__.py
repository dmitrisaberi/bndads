import fire
from experiments import plot_results
from experiments import run_experiments


class Experiments:
    def plot_experiments(self):
        plot_results.main()
    
    def run_experiments(self):
        run_experiments.main()
    
    def run_and_plot(self):
        self.run_experiments()
        self.plot_experiments()

if __name__ == "__main__":
    fire.Fire(Experiments)