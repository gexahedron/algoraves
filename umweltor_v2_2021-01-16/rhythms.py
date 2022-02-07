import random

# code from https://github.com/brianhouse/bjorklund/blob/master/__init__.py
def euclidean(steps, pulses):
    steps = int(steps)
    pulses = int(pulses)
    if pulses > steps:
        raise ValueError    
    pattern = []    
    counts = []
    remainders = []
    divisor = steps - pulses
    remainders.append(pulses)
    level = 0
    while True:
        counts.append(divisor // remainders[level])
        remainders.append(divisor % remainders[level])
        divisor = remainders[level]
        level = level + 1
        if remainders[level] <= 1:
            break
    counts.append(divisor)
    
    def build(level):
        if level == -1:
            pattern.append(0)
        elif level == -2:
            pattern.append(1)         
        else:
            for i in range(0, counts[level]):
                build(level - 1)
            if remainders[level] != 0:
                build(level - 2)
    
    build(level)
    i = pattern.index(1)
    pattern = pattern[i:] + pattern[0:i]
    return pattern

def create_rhythm(beat_count, subbeat_count):
    rhythm = []
    for i in range(beat_count):
        note_count = random.randint(1, subbeat_count)
        subrhythm = euclidean(subbeat_count, note_count)
        shift = random.randint(0, subbeat_count - 1)
        subrhythm = subrhythm[shift:] + subrhythm[:shift]
        rhythm += subrhythm
    return rhythm

def create_rhythm2(beat_count, subbeat_count, mode):
    rhythm = []
    total_count = beat_count * subbeat_count
    note_count = random.randint(beat_count, total_count)
    rhythm = euclidean(total_count, note_count)
    shift = random.randint(0, subbeat_count - 1)
    if mode == 'kick':
        shift = 0
    rhythm = rhythm[shift:] + rhythm[:shift]
    return rhythm
