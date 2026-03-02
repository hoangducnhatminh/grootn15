import h5py
import numpy as np

def explore_hdf5(file_path):
    """
    Explore an HDF5 file structure and print its contents.
    
    Args:
        file_path: Path to the HDF5 file
    """
    
    def print_attrs(name, obj):
        """Print attributes of HDF5 objects"""
        indent = '  ' * name.count('/')
        if isinstance(obj, h5py.Group):
            print(f'{indent}📁 Group: {name}')
            if obj.attrs:
                for key, val in obj.attrs.items():
                    print(f'{indent}   @{key}: {val}')
        elif isinstance(obj, h5py.Dataset):
            shape = obj.shape
            dtype = obj.dtype
            print(f'{indent}📊 Dataset: {name}')
            print(f'{indent}   Shape: {shape}, Dtype: {dtype}')
            
            # Print attributes
            if obj.attrs:
                for key, val in obj.attrs.items():
                    print(f'{indent}   @{key}: {val}')
            
            # Print values for small datasets
            if len(shape) == 0 or (len(shape) == 1 and shape[0] == 1):
                print(f'{indent}   Value: {obj[()]}')
            elif np.prod(shape) <= 10:
                print(f'{indent}   Values: {obj[()]}')
    
    print('=' * 80)
    print(f'HDF5 File: {file_path}')
    print('=' * 80)
    
    with h5py.File(file_path, 'r') as f:
        # Print root attributes
        if f.attrs:
            print(f'\n📦 Root attributes:')
            for key, val in f.attrs.items():
                print(f'   @{key}: {val}')
        
        # Print structure
        print('\n' + '=' * 80)
        print('File Structure:')
        print('=' * 80)
        f.visititems(print_attrs)
        
        # Print top-level keys
        print('\n' + '=' * 80)
        print('Top-level keys:')
        print('=' * 80)
        for key in f.keys():
            print(f'  - {key}')
        
        # Explore data structure if it exists
        if 'data' in f:
            print('\n' + '=' * 80)
            print('Data Structure Details:')
            print('=' * 80)
            data_group = f['data']
            episode_keys = list(data_group.keys())
            print(f'Number of episodes: {len(episode_keys)}')
            
            if episode_keys:
                # Show first episode in detail
                sample_ep = episode_keys[0]
                print(f'\nSample episode: {sample_ep}')
                print('Episode contents:')
                for key in data_group[sample_ep].keys():
                    item = data_group[sample_ep][key]
                    if isinstance(item, h5py.Dataset):
                        print(f'  - {key}: shape={item.shape}, dtype={item.dtype}')
                        # Show sample data for small arrays
                        if item.shape[0] <= 3:
                            print(f'    Sample: {item[()]}')
                        elif len(item.shape) > 0:
                            print(f'    First element: {item[0]}')
                
                # Show statistics if there are multiple episodes
                if len(episode_keys) > 1:
                    print(f'\nAll episodes: {episode_keys[:10]}{"..." if len(episode_keys) > 10 else ""}')


def inspect_episode(file_path, episode_key):
    """
    Inspect a specific episode in detail.
    
    Args:
        file_path: Path to the HDF5 file
        episode_key: Key of the episode to inspect (e.g., 'episode_0')
    """
    print('=' * 80)
    print(f'Episode: {episode_key}')
    print('=' * 80)
    
    with h5py.File(file_path, 'r') as f:
        if 'data' not in f or episode_key not in f['data']:
            print(f'Episode {episode_key} not found!')
            return
        
        episode = f['data'][episode_key]
        
        for key in episode.keys():
            item = episode[key]
            if isinstance(item, h5py.Dataset):
                print(f'\n{key}:')
                print(f'  Shape: {item.shape}')
                print(f'  Dtype: {item.dtype}')
                print(f'  Min: {np.min(item[()])}')
                print(f'  Max: {np.max(item[()])}')
                print(f'  Mean: {np.mean(item[()])}')
                if item.shape[0] <= 5:
                    print(f'  Data:\n{item[()]}')
                else:
                    print(f'  First element: {item[0]}')
                    print(f'  Last element: {item[-1]}')


if __name__ == '__main__':
    file_path = 'dataset/dataset_annotated_gr1.hdf5'
    
    # Explore the full structure
    explore_hdf5(file_path)
    
    # Uncomment to inspect a specific episode
    # inspect_episode(file_path, 'episode_0')
