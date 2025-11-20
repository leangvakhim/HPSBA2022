import matplotlib.pyplot as plt
from wsn import wsn
from hpsba import hpsba
area_width = 100
area_height = 100
num_nodes = 45
r_sensing = 10.0
max_iter = 150
population = 30
resolution=1.0

wsn_env = wsn(area_width, area_height, num_nodes, r_sensing, resolution)
dim = num_nodes * 2
bounds = (0, area_width)

optimizer = hpsba(wsn_env.evaluate, bounds, dim, population, max_iter)

best_pos, best_score, curve = optimizer.run()
final_coverage = (1.0 - float(best_score)) * 100
print(f"Final Coverage: {final_coverage:.2f}%")
wsn_env.plot(best_pos, title=f"HPSBA Coverage: {final_coverage:.2f}%")

plt.figure()
coverage_curve = [(1.0 - float(score)) * 100 for score in curve]
plt.plot(coverage_curve, 'b-', label='HPSBA Coverage')
plt.xlabel('Iteration')
plt.ylabel('Coverage (%)')
plt.title('HPSBA Coverage Curve')
plt.legend()
plt.show()
