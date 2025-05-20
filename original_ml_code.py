# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load preprocessed dataset
df = pd.read_csv("processed_data (1).csv" )

# Extract features and target
X_mRNA = df.iloc[:, 1:121].values  # One-hot encoded mRna (121 columns)
X_gene = df.iloc[:, 121:138].values  # One-hot encoded GENE sequences (17 columns)
X_position = df["Amino_Acid_Cut_Position"].values.reshape(-1, 1)  # Single column for position
y = df["predictions"].values  # Target efficiency

# Combine all features into a single input matrix
X = np.hstack((X_mRNA, X_gene, X_position))

# Print feature distribution
print(f" Target Gene Features: {X_gene.shape[1]}")
print(f" mRNA Sequence Features: {X_mRNA.shape[1]}")
print(f" Amino Acid Cut Position Features: {X_position.shape[1]}")
print(f" Total Features Used: {X.shape[1]}")

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n Model Evaluation Metrics:")
print(f" Mean Squared Error (MSE): {mse:.4f}")
print(f" Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f" Mean Absolute Error (MAE): {mae:.4f}")
print(f" R2 Score: {r2:.4f}")

# Get unique target genes from the dataset for user selection
unique_target_genes = []
for i in range(X_gene.shape[1]):
    # Find samples where this gene is active (1)
    samples = df.iloc[np.where(X_gene[:, i] == 1)[0]]
    if len(samples) > 0:
        # Get the gene name - assuming it's in the dataset somewhere
        # This is a placeholder - modify to extract the actual gene names from your dataset
        gene_name = f"Gene_{i+1}"
        unique_target_genes.append((gene_name, i))

# Function to convert nucleotide sequence to one-hot encoding
def sequence_to_one_hot(sequence):
    """
    Convert a nucleotide sequence (A, C, G, T) to one-hot encoding.
    Returns a 120-length vector for a 30-nucleotide sequence.
    """
    # Check sequence length
    if len(sequence) != 30:
        raise ValueError("mRNA sequence must be exactly 30 nucleotides long.")

    # Validate sequence characters
    valid_nucleotides = {'A', 'C', 'G', 'T'}
    if not all(n in valid_nucleotides for n in sequence.upper()):
        raise ValueError("mRNA sequence must contain only A, C, G, T nucleotides.")

    # Create one-hot encoding
    sequence = sequence.upper()
    one_hot = []

    for nucleotide in sequence:
        if nucleotide == 'A':
            one_hot.extend([1, 0, 0, 0])
        elif nucleotide == 'C':
            one_hot.extend([0, 1, 0, 0])
        elif nucleotide == 'G':
            one_hot.extend([0, 0, 1, 0])
        elif nucleotide == 'T':
            one_hot.extend([0, 0, 0, 1])

    return one_hot

# Function to create one-hot encoding for target gene
def target_gene_to_one_hot(gene_index, num_genes=17):
    """
    Create a one-hot encoded vector for the target gene.
    """
    one_hot = [0] * num_genes
    one_hot[gene_index] = 1
    return one_hot

# Prediction function (unchanged from your code)
def predict_efficiency(mrna_sequence, target_gene, amino_acid_cut_position):
    """
    Predicts efficiency based on given mRNA sequence, target gene, and cut position.
    - `mrna_sequence`: One-hot encoded (120-length) mRNA sequence.
    - `target_gene`: One-hot encoded (17-length) target gene.
    - `amino_acid_cut_position`: Numerical value.
    """
    if len(mrna_sequence) != 120:
        raise ValueError("mRNA sequence must be a one-hot encoded vector of length 120.")
    if len(target_gene) != 17:
        raise ValueError("Target gene must be a one-hot encoded vector of length 17.")

    # Convert inputs to 2D array format
    mrna_sequence = np.array(mrna_sequence).reshape(1, -1)
    target_gene = np.array(target_gene).reshape(1, -1)
    amino_acid_cut_position = np.array([[amino_acid_cut_position]])  # Ensure it's 2D

    # Combine features
    input_features = np.hstack((mrna_sequence, target_gene, amino_acid_cut_position))

    # Predict efficiency
    predicted_efficiency = model.predict(input_features)[0]
    return predicted_efficiency

# Function to predict efficiency at multiple cut positions
def predict_efficiency_at_positions(mrna_sequence, target_gene, positions):
    """
    Predicts efficiency at multiple amino acid cut positions.
    Returns a list of (position, efficiency) tuples.
    """
    results = []
    for pos in positions:
        eff = predict_efficiency(mrna_sequence, target_gene, pos)
        results.append((pos, eff))
    return results

# User input interface
def get_user_input():
    print("\n CRISPR Efficiency Predictor ")
    print("================================")

    # Get mRNA sequence input
    print("\nEnter the mRNA sequence (30 nucleotides, A/C/G/T only):")
    mrna_input = input("> ")

    # Display target gene options
    print("\nSelect target gene by entering the corresponding number:")
    for i, (gene_name, _) in enumerate(unique_target_genes):
        print(f"{i+1}. {gene_name}")

    # Get target gene selection
    while True:
        try:
            gene_selection = int(input("> ")) - 1
            if 0 <= gene_selection < len(unique_target_genes):
                selected_gene = unique_target_genes[gene_selection]
                break
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a number.")

    # Get amino acid cut position
    print("\nEnter the amino acid cut position:")
    position_input = input("> ")

    try:
        # Process inputs
        mrna_one_hot = sequence_to_one_hot(mrna_input)
        gene_one_hot = target_gene_to_one_hot(selected_gene[1])

        # Check if specific position or multiple positions
        if position_input.strip().lower() == "all":
            # Predict efficiency for multiple positions (from 1 to 30)
            positions = range(1, 31)
            results = predict_efficiency_at_positions(mrna_one_hot, gene_one_hot, positions)

            # Find the position with maximum efficiency
            max_pos, max_eff = max(results, key=lambda x: x[1])

            print("\n Efficiency predictions by cut position:")
            print("Position  |  Efficiency")
            print("-----------------------")
            for pos, eff in sorted(results):
                star = " " if pos == max_pos else ""
                print(f"{pos:8d}  |  {eff:.4f}{star}")

            print(f"\n Optimal cut position: {max_pos} with efficiency {max_eff:.4f}")

        else:
            try:
                amino_acid_cut_position = int(position_input)
                if amino_acid_cut_position < 1 or amino_acid_cut_position > 30:
                    print("Warning: Position outside typical range (1-30). Proceeding anyway.")

                # Make prediction for the specific position
                efficiency = predict_efficiency(mrna_one_hot, gene_one_hot, amino_acid_cut_position)

                print(f"\n Predicted Efficiency at position {amino_acid_cut_position}: {efficiency:.4f}")

                # Add interpretation of the efficiency score
                if efficiency > 0.75:
                    print("Interpretation: High efficiency expected")
                elif efficiency > 0.5:
                    print("Interpretation: Moderate efficiency expected")
                else:
                    print("Interpretation: Low efficiency expected")

                # Show nearby positions as well
                print("\nEfficiency at nearby positions:")
                nearby = range(max(1, amino_acid_cut_position-2), min(1997, amino_acid_cut_position+3))
                nearby_results = predict_efficiency_at_positions(mrna_one_hot, gene_one_hot, nearby)

                for pos, eff in nearby_results:
                    marker = "->" if pos == amino_acid_cut_position else " "
                    print(f"{marker} Position {pos}: {eff:.4f}")

            except ValueError:
                print("Invalid position. Please enter a number.")
                return

    except ValueError as e:
        print(f"Error: {e}")

# Show example usage first (using your original example)
sample_mrna = X_mRNA[0]  # Example: Take first mRNA sample
sample_target_gene = X_gene[0]  # Example: Take first target gene
sample_cut_position = X_position[0][0]  # Example: Take first amino acid cut position
predicted_efficiency = predict_efficiency(sample_mrna, sample_target_gene, sample_cut_position)
print(f"\n Example Prediction (using first sample): {predicted_efficiency:.4f}")

# Run the user input interface
if __name__ == "__main__":
    get_user_input()