import numpy as np
import matplotlib.pyplot as plt

# Use -1 for out of bounds, 0 for starting points, 1 for track, 2 for finish line

def build_track_a(save = False):
    track = np.ones((32,17))
    track[14:, 0] = -1
    track[22:, 1] = -1
    track[-3:, 2] = -1
    track[:4, 0] = -1
    track[:3, 1] = -1
    track[0, 2] = -1
    track[6:, -8:] = -1

    track[6, 9] = 1

    track[:6, -1] = 2
    track[-1, 3:9] = 0

    if save:
        with open('./5.12/tracks/track_a.npy', 'wb') as f:
            np.save(f, track)

    return track

def build_track_b(save = False):
    track = np.ones(shape=(30, 32))

    for i in range(14):
        track[:(-3 - i), i] = -1
    
    track[3:7, 11] = 1
    track[2:8, 12] = 1
    track[1:9, 13] = 1
   
    track[0, 14:16] = -1
    track[-17:, -9:] = -1
    track[12, -8:] = -1
    track[11, -6:] = -1
    track[10, -5:] = -1
    track[9, -2:] = -1

    track[-1, track[-1] != -1] = 0
    track[track[:, -1] != -1, -1] = 2


    if save:
        with open('./5.12/tracks/track_b.npy', 'wb') as f:
            np.save(f, track)

    return track

if __name__ == "__main__":
    track_a = build_track_a(save = True)
    track_b = build_track_b(save = True)

    plt.figure(figsize=(10, 5))
    for i, map_type in enumerate(['a', 'b']): # track a or track b?
        with open(f'./5.12/tracks/track_{map_type}.npy', 'rb') as f:
            track = np.load(f)

        plt.subplot(1, 2, i + 1)
        plt.imshow(track, cmap='GnBu')
        plt.title(f'Track {map_type}')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('./5.12/tracks/tracks.png')
    plt.show()