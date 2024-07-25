import os
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f'{current_dir}/restaurant.txt')

file_path = os.path.join(current_dir, 'restaurant.txt')
print(f'2. Full file path: {file_path}')
