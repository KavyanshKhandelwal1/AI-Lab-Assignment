import numpy as np
from PIL import Image
import random
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

class JigsawSolver:
    """
    Solves a jigsaw puzzle using Simulated Annealing.
    """

    def _init_(self, image_path, patch_size):
        """
        Initializes the solver.
        Args:
            image_path (str): Path to the image file.
            patch_size (int): The size (width and height) of each square puzzle piece.
        """
        self.image_path = image_path
        self.patch_size = patch_size
        self.patches, self.grid_shape = self._create_patches()
        self.num_patches = len(self.patches)

    def _create_patches(self):
        """
        Loads an image, crops it to be divisible by patch_size, 
        and splits it into a list of patches.
        """
        img = Image.open(self.image_path)

        # Crop image to be divisible by patch size
        width, height = img.size
        new_width = (width // self.patch_size) * self.patch_size
        new_height = (height // self.patch_size) * self.patch_size
        img = img.crop((0, 0, new_width, new_height))

        patches = []
        grid_shape = (new_height // self.patch_size, new_width // self.patch_size)
        
        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                box = (j * self.patch_size, i * self.patch_size, (j + 1) * self.patch_size, (i + 1) * self.patch_size)
                patch = img.crop(box)
                patches.append(np.array(patch))
                
        return patches, grid_shape

    def _calculate_dissimilarity(self, patch1, patch2, orientation='H'):
        """
        Calculates the dissimilarity (energy) between two adjacent patches.
        Uses Sum of Squared Differences (SSD) on the meeting edges.
        """
        if orientation == 'H':  # Horizontal adjacency
            # Compare right edge of patch1 with left edge of patch2
            return np.sum((patch1[:, -1] - patch2[:, 0]) ** 2)
        else:  # Vertical adjacency
            # Compare bottom edge of patch1 with top edge of patch2
            return np.sum((patch1[-1, :] - patch2[0, :]) ** 2)

    def calculate_total_energy(self, state):
        """
        Calculates the total energy of the current puzzle arrangement.
        Energy is the sum of dissimilarities of all adjacent patches.
        """
        total_energy = 0
        rows, cols = self.grid_shape
        
        # Reshape the flat list of patches into a grid for easier processing
        grid = np.array(state).reshape(rows, cols, self.patch_size, self.patch_size, 3)

        # Horizontal dissimilarities
        for i in range(rows):
            for j in range(cols - 1):
                total_energy += self._calculate_dissimilarity(grid[i, j], grid[i, j+1], 'H')
        
        # Vertical dissimilarities
        for i in range(rows - 1):
            for j in range(cols):
                total_energy += self._calculate_dissimilarity(grid[i, j], grid[i+1, j], 'V')
                
        return total_energy

    def solve(self, initial_temp=10000, cooling_rate=0.999, min_temp=1.0, max_iter=50000):
        """
        The main Simulated Annealing loop to solve the puzzle.
        """
        # 1. Initialization
        current_state = list(self.patches)
        random.shuffle(current_state)
        
        current_energy = self.calculate_total_energy(current_state)
        
        best_state = list(current_state)
        best_energy = current_energy
        
        temp = initial_temp
        
        # Use tqdm for a progress bar
        pbar = tqdm(total=max_iter, desc="Solving Puzzle")

        for i in range(max_iter):
            if temp <= min_temp:
                break
                
            # 2. Generate a neighbor state by swapping two random patches
            idx1, idx2 = random.sample(range(self.num_patches), 2)
            
            new_state = list(current_state)
            new_state[idx1], new_state[idx2] = new_state[idx2], new_state[idx1]
            
            new_energy = self.calculate_total_energy(new_state)
            
            # 3. Acceptance criterion
            delta_energy = new_energy - current_energy
            
            if delta_energy < 0 or random.random() < math.exp(-delta_energy / temp):
                current_state = new_state
                current_energy = new_energy
            
            # Keep track of the best solution found so far
            if current_energy < best_energy:
                best_state = list(current_state)
                best_energy = current_energy

            # 4. Cooling
            temp *= cooling_rate
            pbar.update(1)
            pbar.set_postfix({"Temp": f"{temp:.2f}", "Best Energy": f"{best_energy:,.0f}"})

        pbar.close()
        print(f"Finished. Best energy found: {best_energy}")
        return best_state, self.get_scrambled_image(current_state)


    def reconstruct_image(self, state):
        """
        Stitches the patches from a given state back into a single image.
        """
        rows, cols = self.grid_shape
        reconstructed = Image.new('RGB', (cols * self.patch_size, rows * self.patch_size))
        
        for i in range(rows):
            for j in range(cols):
                patch_index = i * cols + j
                patch_data = state[patch_index]
                patch_image = Image.fromarray(patch_data.astype('uint8'))
                reconstructed.paste(patch_image, (j * self.patch_size, i * self.patch_size))
                
        return reconstructed

    def get_scrambled_image(self, initial_state):
        """Helper to get the initial scrambled image."""
        return self.reconstruct_image(initial_state)

def create_dummy_image(path="dummy_image.png", width=256, height=256):
    """Creates a simple gradient image for demonstration."""
    if os.path.exists(path):
        return
    array = np.zeros((height, width, 3), dtype=np.uint8)
    for x in range(width):
        for y in range(height):
            array[y, x, 0] = x % 256  # Red gradient
            array[y, x, 1] = y % 256  # Green gradient
            array[y, x, 2] = (x+y) % 256 # Blue gradient
    img = Image.fromarray(array)
    img.save(path)
    print(f"Created dummy image at {path}")

if _name_ == '_main_':
    # --- Configuration ---
    IMAGE_FILE = "dummy_image.png"
    PATCH_SIZE = 32  # Size of each square puzzle piece
    
    # SA Parameters
    INITIAL_TEMP = 1e5
    COOLING_RATE = 0.999
    MIN_TEMP = 0.1
    MAX_ITER = 20000

    # --- Main Execution ---
    # Create a dummy image if the target image doesn't exist
    create_dummy_image(IMAGE_FILE, 256, 256)

    # 1. Initialize the solver
    solver = JigsawSolver(IMAGE_FILE, PATCH_SIZE)
    
    # 2. Start the solving process
    solved_state, scrambled_image = solver.solve(
        initial_temp=INITIAL_TEMP,
        cooling_rate=COOLING_RATE,
        min_temp=MIN_TEMP,
        max_iter=MAX_ITER
    )
    
    # 3. Reconstruct the final image from the best state found
    solved_image = solver.reconstruct_image(solved_state)
    original_image = solver.reconstruct_image(solver.patches)

    # 4. Display results
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(scrambled_image)
    axes[1].set_title('Scrambled Puzzle')
    axes[1].axis('off')

    axes[2].imshow(solved_image)
    axes[2].set_title('Solved Puzzle')
    axes[2].axis('off')
    
    plt.suptitle("Jigsaw Puzzle Solver using Simulated Annealing")
    plt.tight_layout()
    plt.show()
