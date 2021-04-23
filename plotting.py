import numpy as np
# Plotting
import matplotlib
import itertools
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.cm as cm

class Plotting(object):
    def set_legend(self):
      legendElements = [
            Line2D([0], [0], linestyle='none',marker='o', color='red',markerfacecolor='red', markersize=9),
            Line2D([0], [0], linestyle='none',marker='o', color='green', markerfacecolor='green', markersize=9),
            Line2D([0], [0], linestyle='-', marker='.', color='black', markerfacecolor='black',markersize=0),
            Line2D([0], [0], linestyle='--', marker='.', color='blue', markerfacecolor='black', markersize=0),
            Line2D([0], [0], linestyle='none', marker='.', color='black', markerfacecolor='black', markersize=9)
      ]
      return legendElements

    def plot_margin(self,X,y,objFit):
        fig = plt.figure()  # create a figure object
        ax = fig.add_subplot(1, 1, 1)
        X1 = X[y == 1]
        X2 = X[y == -1]
        y1 = y[y == 1]
        y2 = y[y == -1]
        grid_size = 200
        # Format plot area:
        ax = plt.gca()
        ax = plt.axes(facecolor='#FFFD03')  # background color.
        # Axis limits.
        x1_min, x1_max = X1.min(), X1.max()
        x2_min, x2_max = X2.min(), X2.max()
        ax.set(xlim=(x1_min, x1_max), ylim=(x2_min, x2_max))
        # Labels
        plt.xlabel('$x_1$', fontsize=9)
        plt.ylabel('$x_2$', fontsize=9)
        legendElements = self.set_legend()
        
        myLegend = plt.legend(legendElements,   ['Negative', 'Positive','Decision Boundary','Margin','Support Vectors'],fontsize="7",loc='lower center', bbox_to_anchor=(0.7, 0.98))
        # plot points
        plt.plot(X1[:, 0], X1[:, 1], marker='o',markersize=5, color='red',linestyle='none')
        plt.plot(X2[:, 0], X2[:, 1], marker='o',markersize=4, color='green',linestyle='none')
        plt.scatter(objFit.sv[:, 0], objFit.sv[:, 1], s=60, color="blue")   # The points designating the support vectors.
        if  objFit.kernel_type  == 'polynomial' or objFit.kernel_type  == 'gaussian':
           # Will use a contour plot.
           x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
           y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
           xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size),
                              np.linspace(y_min, y_max, grid_size),
                              indexing='ij')
           flatten = lambda m: np.array(m).reshape(-1,)

           result = []
           for (i, j) in itertools.product(range(grid_size), range(grid_size)):
              point = np.array([xx[i, j], yy[i, j]]).reshape(1, 2)
              result.append(objFit.predict(point))

           Z = np.array(result).reshape(xx.shape)
           plt.clf()
           plt.contourf(xx, yy, Z,
                 cmap=cm.tab20,
                 levels=[-0.0001, 0.0001],
                 extend='both',
                 alpha=0.4)
           plt.scatter(flatten(X[:, 0]), flatten(X[:, 1]),
                c=flatten(y), cmap=cm.tab20)
           plt.xlim(x_min, x_max)
           plt.ylim(y_min, y_max)
          
        else:
            # Linear margin line needs to be generated.
            w = objFit.w
            
            c = objFit.b
            _y1 = (-w[0] * x1_min - c ) / w[1]
            _y2 = (-w[0] * x1_max - c ) / w[1]
            plt.plot([x1_min, x1_max], [_y1, _y2], "k")

            #upper margin
            _y3 = (-w[0] * x1_min - c + 1) / w[1]
            _y4 = (-w[0] * x1_max - c  + 1) / w[1]
            plt.plot([x1_min, x1_max], [_y3, _y4], "k--")

            #lower_argin
            _y5 = (-w[0] * x1_min - c - 1 ) / w[1]
            _y6 = (-w[0] * x1_max - c - 1 ) / w[1]
            plt.plot([x1_min, x1_max], [_y5, _y6], "k--")
        plt.show(block=False)

