import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any
from utils.logging_config import get_logger
from hydra.core.hydra_config import HydraConfig
import os


class ReportGenerator:
    
    def __init__(self, config):
        self.config = config
        self.logger = get_logger()
        self.results = []
        
    def generate_comparison_report(self) -> Dict[str, Any]:
        self._load_results()
        
        if len(self.results) < 2:
            self.logger.warning("Need at least 2 experiments for comparison")
            return {"output_dir": None, "files": {}}
        
        hydra_output_dir = HydraConfig.get().runtime.output_dir
        output_dir = Path(hydra_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        files = {}
        
        # Create comparison table with rounded values
        df = self._create_comparison_table()
        
        # Save CSV table
        table_file = output_dir / "comparison_table.csv"
        df.to_csv(table_file, index=False)
        files["table"] = str(table_file)
        
        # Create combined report image (table + chart)
        combined_file = output_dir / "comparison_report.png"
        self._create_combined_report(df, str(combined_file))
        files["combined_report"] = str(combined_file)
        
        # Also create standalone chart
        chart_file = output_dir / "comparison_chart.png"
        self._create_bar_chart(df, str(chart_file))
        files["chart"] = str(chart_file)
        
        self.logger.info(f"Report generated in: {output_dir}")
        
        return {"output_dir": str(output_dir), "files": files}
    
    def _load_results(self):
        for input_dir in self.config.input_dirs:
            results_file = Path(input_dir) / "results.json"
            if results_file.exists():
                with open(results_file) as f:
                    data = json.load(f)
                    self.results.append(data)
                    self.logger.info(f"Loaded results from: {results_file}")
            else:
                self.logger.warning(f"Results file not found: {results_file}")
    
    def _create_comparison_table(self) -> pd.DataFrame:
        rows = []
        for result in self.results:
            row = {
                "Experiment": result["experiment_name"],
                "Model": result["model_name"],
                "Dataset": result["dataset_name"]
            }
            # Round all metric values to 4 decimal places
            for metric_name, value in result["results"].items():
                row[metric_name] = round(value, 4)
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _create_combined_report(self, df: pd.DataFrame, save_path: str):
        """Create a combined image with both table and chart"""
        metric_cols = [col for col in df.columns if col not in ["Experiment", "Model", "Dataset"]]
        
        if not metric_cols:
            return
        
        # Create figure with subplots (table on top, chart on bottom)
        fig = plt.figure(figsize=(14, 10))
        
        # Chart subplot (bottom)
        ax_chart = plt.subplot2grid((3, 1), (1, 0), rowspan=2)
        
        # Add more horizontal spacing between experiment groups
        x_spacing = 1.5  # Increase spacing between experiments
        x = [i * x_spacing for i in range(len(df))]
        width = 0.8 / len(metric_cols)
        
        for i, metric in enumerate(metric_cols):
            ax_chart.bar([pos + i * width for pos in x], df[metric], width, label=metric)
        
        ax_chart.set_xlabel("Experiments")
        ax_chart.set_ylabel("Score")
        ax_chart.set_title(self.config.title)
        ax_chart.set_xticks([pos + width * (len(metric_cols) - 1) / 2 for pos in x])
        ax_chart.set_xticklabels(df["Experiment"], rotation=45)
        ax_chart.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax_chart.grid(True, alpha=0.3)
        
        # Table subplot (top)
        ax_table = plt.subplot2grid((3, 1), (0, 0))
        ax_table.axis('off')
        
        # Create table data for display (rounded values)
        table_data = []
        for _, row in df.iterrows():
            table_row = [row["Experiment"]]
            for metric in metric_cols:
                table_row.append(f"{row[metric]:.4f}")
            table_data.append(table_row)
        
        # Create the table
        col_labels = ["Experiment"] + metric_cols
        table = ax_table.table(cellText=table_data,
                              colLabels=col_labels,
                              cellLoc='center',
                              loc='center',
                              bbox=[0, 0, 1, 1])
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Color header row
        for i in range(len(col_labels)):
            table[(0, i)].set_facecolor('#E6E6FA')
            table[(0, i)].set_text_props(weight='bold')
        
        # Color data rows alternately
        for i in range(1, len(table_data) + 1):
            for j in range(len(col_labels)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#F8F8FF')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Combined report saved to: {save_path}")
    
    def _create_bar_chart(self, df: pd.DataFrame, save_path: str):
        """Create standalone bar chart"""
        metric_cols = [col for col in df.columns if col not in ["Experiment", "Model", "Dataset"]]
        
        if not metric_cols:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Add more horizontal spacing between experiment groups
        x_spacing = 1.5  # Increase spacing between experiments
        x = [i * x_spacing for i in range(len(df))]
        width = 0.8 / len(metric_cols)
        
        for i, metric in enumerate(metric_cols):
            ax.bar([pos + i * width for pos in x], df[metric], width, label=metric)
        
        ax.set_xlabel("Experiments")
        ax.set_ylabel("Score")
        ax.set_title(self.config.title)
        ax.set_xticks([pos + width * (len(metric_cols) - 1) / 2 for pos in x])
        ax.set_xticklabels(df["Experiment"], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Chart saved to: {save_path}") 