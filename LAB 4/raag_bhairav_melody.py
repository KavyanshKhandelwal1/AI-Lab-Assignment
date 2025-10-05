import random
import numpy as np
from scipy.io.wavfile import write
from tqdm import tqdm

# --- Musical Configuration for Raag Bhairav ---

# Notes in Raag Bhairav (r and d are komal/flat)
# S' is the Sa of the next octave
BHAIRAV_SWARAS = ['S', 'r', 'G', 'M', 'P', 'd', 'N', 'S\'']

# Vadi (King note) and Samvadi (Queen note)
VADI = 'd'
SAMVADI = 'r'

# Characteristic phrases (Pakar) that define the Raag's identity
BHAIRAV_PHRASES = [
    ['G', 'M', 'd', 'P'],
    ['G', 'M', 'r', 'S'],
    ['d', 'N', 'S\''],
    ['M', 'P', 'd', 'P', 'M', 'G', 'M']
]

# Note-to-integer mapping for calculating melodic distance
NOTE_MAP = {note: i for i, note in enumerate(BHAIRAV_SWARAS)}

# --- Genetic Algorithm Parameters ---
POPULATION_SIZE = 100
MELODY_LENGTH = 16  # Length of the melody in notes
MUTATION_RATE = 0.05
CROSSOVER_RATE = 0.8
NUM_GENERATIONS = 200

# --- Audio Generation Parameters ---
TONIC_FREQ = 261.63  # Frequency of Sa (C4)
NOTE_DURATION = 0.3  # seconds per note
SAMPLE_RATE = 44100  # CD quality audio

class MelodyGenerator:
    """
    Generates a melody in Raag Bhairav using a Genetic Algorithm.
    """

    def _calculate_fitness(self, melody):
        """
        The fitness function. Scores a melody based on Raag Bhairav rules.
        """
        score = 0

        # Rule 1: Vadi/Samvadi Emphasis (+15 for Vadi, +10 for Samvadi)
        score += melody.count(VADI) * 15
        score += melody.count(SAMVADI) * 10

        # Rule 2: Presence of Characteristic Phrases (High reward)
        melody_str = " ".join(melody)
        for phrase in BHAIRAV_PHRASES:
            phrase_str = " ".join(phrase)
            if phrase_str in melody_str:
                score += 150

        # Rule 3: Smooth Melodic Movement (Chalan)
        for i in range(len(melody) - 1):
            note1 = melody[i]
            note2 = melody[i+1]
            # Calculate distance between notes
            dist = abs(NOTE_MAP[note1] - NOTE_MAP[note2])

            if dist == 0: # Penalize repeated notes
                score -= 10
            elif dist <= 2: # Reward stepwise motion
                score += 5
            elif dist >= 4: # Penalize large jumps
                score -= 20

        # Rule 4: Resolution to Sa (Reward for ending on the tonic)
        if melody[-1] == 'S':
            score += 75

        return max(0, score) # Fitness cannot be negative

    def _create_individual(self):
        """Creates a single random melody."""
        return [random.choice(BHAIRAV_SWARAS) for _ in range(MELODY_LENGTH)]

    def _selection(self, population, fitnesses):
        """Tournament selection: Pick 3 random individuals, return the best one."""
        tournament = random.sample(list(zip(population, fitnesses)), k=3)
        tournament.sort(key=lambda x: x[1], reverse=True)
        return tournament[0][0]

    def _crossover(self, parent1, parent2):
        """Single-point crossover."""
        if random.random() < CROSSOVER_RATE:
            point = random.randint(1, MELODY_LENGTH - 1)
            child1 = parent1[:point] + parent2[point:]
            child2 = parent2[:point] + parent1[point:]
            return child1, child2
        return parent1, parent2

    def _mutate(self, melody):
        """Randomly change a note in the melody."""
        for i in range(MELODY_LENGTH):
            if random.random() < MUTATION_RATE:
                melody[i] = random.choice(BHAIRAV_SWARAS)
        return melody

    def evolve(self):
        """
        The main evolution loop of the Genetic Algorithm.
        """
        # 1. Initialize Population
        population = [self._create_individual() for _ in range(POPULATION_SIZE)]

        best_melody_overall = None
        best_fitness_overall = -1

        pbar = tqdm(range(NUM_GENERATIONS), desc="Evolving Melody")
        for gen in pbar:
            # 2. Calculate Fitness
            fitnesses = [self._calculate_fitness(ind) for ind in population]

            # Track the best melody found so far
            best_fitness_current_gen = max(fitnesses)
            if best_fitness_current_gen > best_fitness_overall:
                best_fitness_overall = best_fitness_current_gen
                best_melody_overall = population[fitnesses.index(best_fitness_current_gen)]

            pbar.set_postfix({
                "Best Fitness": f"{best_fitness_overall}",
                "Best Melody": f"{' '.join(best_melody_overall)}"
            })

            # 3. Create New Generation
            new_population = []
            while len(new_population) < POPULATION_SIZE:
                # Selection
                parent1 = self._selection(population, fitnesses)
                parent2 = self._selection(population, fitnesses)
                # Crossover
                child1, child2 = self._crossover(parent1, parent2)
                # Mutation
                new_population.append(self._mutate(child1))
                if len(new_population) < POPULATION_SIZE:
                    new_population.append(self._mutate(child2))
            
            population = new_population
        
        return best_melody_overall


def save_melody_as_wav(melody, filename="raag_bhairav_melody.wav"):
    """
    Converts a melody (list of notes) into a playable WAV file.
    """
    # Frequencies relative to the tonic (Sa) based on the 12-tone equal temperament scale
    # S, r, R, g, G, M, m, P, d, D, n, N
    semitone_ratios = {
        'S': 1.0,           # Tonic
        'r': 2**(1/12),     # Komal Re
        'G': 2**(4/12),     # Shuddha Ga
        'M': 2**(5/12),     # Shuddha Ma
        'P': 2**(7/12),     # Pancham
        'd': 2**(8/12),     # Komal Dha
        'N': 2**(11/12),    # Shuddha Ni
        'S\'': 2.0          # Upper Tonic
    }

    # Generate audio waveform
    waveform = []
    for note in melody:
        freq = TONIC_FREQ * semitone_ratios[note]
        t = np.linspace(0., NOTE_DURATION, int(NOTE_DURATION * SAMPLE_RATE), endpoint=False)
        # Apply a simple fade-out to reduce clicking between notes
        fade_out = np.linspace(1., 0., len(t))**0.5
        sine_wave = np.sin(2. * np.pi * freq * t) * fade_out
        waveform.extend(sine_wave)

    # Normalize to 16-bit PCM format and write to file
    waveform_normalized = np.int16((np.array(waveform) / np.max(np.abs(waveform))) * 32767)
    write(filename, SAMPLE_RATE, waveform_normalized)
    print(f"\n✅ Successfully saved melody to {filename}")


if _name_ == '_main_':
    generator = MelodyGenerator()
    best_melody = generator.evolve()
    
    print("\n--- Generation Complete ---")
    print(f"Best melody found: {' '.join(best_melody)}")
    print(f"Final Fitness Score: {generator._calculate_fitness(best_melody)}")
    
    save_melody_as_wav(best_melody)
