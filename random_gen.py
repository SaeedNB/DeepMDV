import torch
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(10)

# Generate 2 random pairs of coordinates in [0, 1]
initial_coords = torch.rand((2, 2))

# Number of additional random points
num_points = 1000

# Generate data for the additional 200 pairs of coordinates

# Normal Distribution (rescaled to [0,1])
normal_data_x = torch.randn(num_points)
normal_data_x = (normal_data_x - normal_data_x.min()) / (normal_data_x.max() - normal_data_x.min())

normal_data_y = torch.randn(num_points)
normal_data_y = (normal_data_y - normal_data_y.min()) / (normal_data_y.max() - normal_data_y.min())

# Gamma Distribution (rescaled to [0,1])
shape = 7
scale = 1
gamma_dist = torch.distributions.Gamma(shape, scale)
gamma_data_x = gamma_dist.sample((num_points,))
gamma_data_x = (gamma_data_x - gamma_data_x.min()) / (gamma_data_x.max() - gamma_data_x.min())

gamma_data_y = gamma_dist.sample((num_points,))
gamma_data_y = (gamma_data_y - gamma_data_y.min()) / (gamma_data_y.max() - gamma_data_y.min())

# Log-normal Distribution (rescaled to [0,1])
mean = 1
std = 0.1
log_normal_dist = torch.distributions.LogNormal(mean, std)
log_normal_data_x = log_normal_dist.sample((num_points,))
log_normal_data_x = (log_normal_data_x - log_normal_data_x.min()) / (log_normal_data_x.max() - log_normal_data_x.min())

log_normal_data_y = log_normal_dist.sample((num_points,))
log_normal_data_y = (log_normal_data_y - log_normal_data_y.min()) / (log_normal_data_y.max() - log_normal_data_y.min())

# Create scatter plots in separate charts

# Plot for Initial Coordinates
plt.figure(figsize=(6, 6))
plt.scatter(initial_coords[:, 0], initial_coords[:, 1], color='black', s=100)
plt.title('Initial Coordinates')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True)
plt.show()

# Plot for Normal Distribution
# plt.figure(figsize=(6, 6))
# plt.scatter(normal_data_x, normal_data_y, color='green', alpha=0.5)
# plt.title('Normal Distribution')
# plt.xlabel('X Coordinate')
# plt.ylabel('Y Coordinate')
# plt.xlim(0, 1)
# plt.ylim(0, 1)
# plt.grid(True)
# plt.show()

# Plot for Gamma Distribution
plt.figure(figsize=(6, 6))
plt.scatter(gamma_data_x, gamma_data_y, color='red', alpha=0.5)
plt.title('Gamma Distribution')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True)
plt.show()

# Plot for Log-normal Distribution
# plt.figure(figsize=(6, 6))
# plt.scatter(log_normal_data_x, log_normal_data_y, color='purple', alpha=0.5)
# plt.title('Log-normal Distribution')
# plt.xlabel('X Coordinate')
# plt.ylabel('Y Coordinate')
# plt.xlim(0, 1)
# plt.ylim(0, 1)
# plt.grid(True)
# plt.show()