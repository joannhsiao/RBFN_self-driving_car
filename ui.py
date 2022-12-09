from rbfn import *
from mlp import *
import tkinter as tk
from matplotlib.patches import Circle, Rectangle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog as fd
from matplotlib.figure import Figure
import os
import math
from matplotlib import animation

class UI(tk.Tk):
	def __init__(self):
		super().__init__()
		self.x = None
		self.y = None
		self.Phi = 90	# car angel: calcalated from mathematic
		self.Theta = 0 	# wheel angel: received from prediction
		self.radius = 6
		self.wheel_max = 40
		self.wheel_min = -40
		self.car_max = 270
		self.car_min = -90
		self.title("RBFNet")
		self.geometry("650x600")
		self.createWidgets()
		self.draw_path()

	def createWidgets(self):
		""" select file """
		tk.Label(text="File name: ", font=('Comic Sans MS', 12)).grid(row=0, column=0)
		self.filename = tk.StringVar()
		print_filename = tk.Label(self, textvariable=self.filename, font=('Comic Sans MS', 12))
		print_filename.grid(row=0, column=1)
		tk.Button(self, text='Select File', font=('Comic Sans MS', 12), command=self.select_file).grid(row=0, column=2)
		tk.Button(master=self, text='Start', font=('Comic Sans MS', 12), command=self.Start).grid(row=0, column=5)
		""" figures """
		self.figure = Figure(figsize=(5,5), dpi=100)
		self.Path = self.figure.add_subplot(111)
		self.Path_plt = FigureCanvasTkAgg(self.figure, self)
		self.Path_plt.get_tk_widget().grid(row=6, column=0, columnspan=4)
		""" exit button """
		tk.Button(master=self, text='Exit', font=('Comic Sans MS', 12), command=self._quit).grid(row=8, column=8)
	
	def draw_path(self):
		""" Path and start point """
		with open("coordinate.txt", "r") as f:
			lines = f.readlines()
		self.wall_x = []
		self.wall_y = []
		self.end_x = []
		self.end_y = []
		for num, line in enumerate(lines):
			xdata = line.split(",")
			if (num > 2):
				self.wall_x.append(int(xdata[0]))
				self.wall_y.append(int(xdata[1]))
			elif (num > 0):
				self.end_x.append(int(xdata[0]))
				self.end_y.append(int(xdata[1]))
			else:
				self.x = int(xdata[0])
				self.y = int(xdata[1])
				self.Phi = float(xdata[2])
				#wheel = Circle((self.x, self.y), 3, fill=False)
		self.Path.plot(self.wall_x, self.wall_y)
		rec = Rectangle((self.end_x[0], self.end_y[0]), abs(self.end_x[1]-self.end_x[0]), -abs(self.end_y[1]-self.end_y[0]))
		self.Path.add_patch(rec)
		#self.Path.add_patch(wheel)
		
	def draw_track(self, i):
		if i < len(self.x_tmp): 
			self.car_x.append(self.x_tmp[i])
			self.car_y.append(self.y_tmp[i])
		self.line.set_data(self.car_x, self.car_y)

	def reset(self):
		self.x = None
		self.y = None
		self.Path.clear()
		self.draw_path()
		self.Phi = 90
		self.Theta = 0
		self.x_tmp = []
		self.y_tmp = []
		self.car_x, self.car_y = [], []

	def Start(self):
		self.reset()
		""" training """
		self.network, dim = process(self.File, 200, 0.01)
		if dim == 5:
			self.clear_file("track6D.txt")
		else:
			self.clear_file("track4D.txt")
		while True:
			""" whether reach end rec or not """
			a, b, c = self.getLine(self.end_x[0], self.end_y[0], self.end_x[1], self.end_y[1])
			if a * self.x + b * self.y + c > 0:
				print("completed.")
				break

			""" detect and record distances & coordinate """
			front = self.detect_distance(self.x, self.y, 'front')
			right = self.detect_distance(self.x, self.y, 'right')
			left = self.detect_distance(self.x, self.y, 'left')

			""" collision test """
			if self.collision(front) or self.collision(right) or self.collision(left):
				print("collision.")
				break
			print("car: ", self.Phi)
			data = [front, right, left]
			if dim == 5:	# 6D data
				data.insert(0, self.y)
				data.insert(0, self.x)
			data = np.reshape(data, (dim, 1))
			
			""" predict the next wheel degree """
			degree = predict(self.network, data).item()
			self.Theta = degree * (self.wheel_max - self.wheel_min) + self.wheel_min
			print("wheel: ", self.Theta)
			#self.Theta %= 360
			if self.Theta > self.wheel_max:
				self.Theta -= self.wheel_max - self.wheel_min

			""" output file """
			if dim == 5:
				with open("track6D.txt", "a") as f:
					f.write(str(round(self.x, 6)) + " " + str(round(self.y, 6)) + " ")
					f.write(str(round(front, 6)) + " " + str(round(right, 6)) + " " + str(round(left, 6)) + " ")
					f.write(str(round(self.Theta, 6)) + "\n")
			else:
				with open("track4D.txt", "a") as f:
					f.write(str(round(front, 6)) + " " + str(round(right, 6)) + " " + str(round(left, 6)) + " ")
					f.write(str(round(self.Theta, 6)) + "\n")
			
			""" move car """
			self.x_tmp.append(self.x)
			self.y_tmp.append(self.y)
			self.Phi = math.radians(self.Phi)
			self.Theta = math.radians(self.Theta)
			self.x += math.cos(self.Phi + self.Theta) + math.sin(self.Theta) * math.sin(self.Phi)
			self.y += math.sin(self.Phi + self.Theta) - math.sin(self.Theta) * math.cos(self.Phi)
			
			#wheel = Circle((self.x, self.y), self.radius/2, fill=False)
			#self.Path.add_patch(wheel)
			
			""" car degree """
			self.Phi -= math.asin(2 * math.sin(self.Theta) / self.radius)
			self.Phi = math.degrees(self.Phi)
			self.Theta = math.degrees(self.Theta)
			
			#self.Phi %= 360
			if self.Phi > self.car_max:
				self.Phi -= self.car_max - self.car_min

		""" draw the track """
		len_path = [i for i in range(len(self.x_tmp))]
		self.line, = self.Path.plot(self.car_x, self.car_y, marker='.', linestyle='', mfc='none')
		anim = animation.FuncAnimation(self.figure, self.draw_track, frames=len_path, interval=len(len_path))	
		self.Path_plt.draw()
			
	def collision(self, distance):
		if (distance <= self.radius/2):
			return True
		else:
			return False

	def detect_distance(self, x, y, point='center'):
		if point == 'right':
			right_angle = self.Phi - 45
			new_x, new_y = self.point_on_circle(x, y, self.radius/2, right_angle)

		elif point == 'left':
			left_angle = self.Phi + 45
			new_x, new_y = self.point_on_circle(x, y, self.radius/2, left_angle)

		elif point == 'front':
			new_x, new_y = self.point_on_circle(x, y, self.radius/2, self.Phi)
		else:
			new_x = x
			new_y = y

		a, b, c = self.getLine(new_x, new_y, x, y)
		inter_x, inter_y = self.WallsIntersection(a, b, c)
		#self.Path.plot([self.x, inter_x], [self.y, inter_y])

		return self.cal_distance(inter_x, inter_y, self.x, self.y)

	def point_on_circle(self, x, y, r, angle):
		rad = math.radians(angle)
		new_x = x + r * math.cos(rad)
		new_y = y + r * math.sin(rad)
		return new_x, new_y

	def getLine(self, x1, y1, x2, y2):
		# ax+by+c
		sign = 1
		a = y2 - y1
		if a < 0:
			sign = -1
			a = sign * a
		b = sign * (x1 - x2)
		c = sign * (y1 * x2 - x1 * y2)
		return a, b, c

	def WallsIntersection(self, a, b, c):
		# line: ax+by+c=0
		inter_x = self.x
		inter_y = self.y
		for i in range(len(self.wall_x)-1):
			a1, b1, c1 = self.getLine(self.wall_x[i], self.wall_y[i], self.wall_x[i+1], self.wall_y[i+1])
			x_, y_ = self.IntersectPoint(a, b, c, a1, b1, c1)
			if self.cal_distance(x_, y_, self.x, self.y) < self.cal_distance(inter_x, inter_y, self.x, self.y):
				if self.intersect_constraint(x_, y_) and self.wall_constrain(self.wall_x[i], self.wall_y[i], self.wall_x[i+1], self.wall_y[i+1], x_, y_):
					inter_x = x_
					inter_y = y_
			else:
				if inter_x == self.x and inter_y == self.y:
					if self.intersect_constraint(x_, y_) and self.wall_constrain(self.wall_x[i], self.wall_y[i], self.wall_x[i+1], self.wall_y[i+1], x_, y_):
						inter_x = x_
						inter_y = y_
		return inter_x, inter_y

	def intersect_constraint(self, x, y):
		# constrain equation F
		left_x, left_y = self.point_on_circle(self.x, self.y, self.radius, self.Phi+90)
		right_x, right_y = self.point_on_circle(self.x, self.y, self.radius, self.Phi-90)
		a, b, c = self.getLine(left_x, left_y, right_x, right_y)
		if a * x + b * y + c < 0:
			return False
		else: 
			return True

	def wall_constrain(self, x1, y1, x2, y2, x_, y_):
		if x_ <= max(x1, x2) and x_ >= min(x1, x2) and y_ <= max(y1, y2) and y_ >= min(y1, y2):
			return True
		else:
			return False

	def cal_distance(self, x1, y1, x2, y2):
		return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

	def IntersectPoint(self, a, b, c, a1, b1, c1):
		# ax+by+c=0 -> ax+by=c
		A = np.array([[a, b], [a1, b1]])
		B = np.array([-c, -c1]).reshape(2, 1)
		ans = np.linalg.solve(A, B)
		return ans[0].item(), ans[1].item()

	def select_file(self):
		filetypes = (('text files', '*.txt'), ('All files', '*.*'))
		self.File = fd.askopenfilename(title='Open a file', initialdir='/', filetypes=filetypes)
		file = ""
		for i in range(len(self.File) - 1, 0, -1):
			if self.File[i] == '/':
				file = self.File[i+1:]
				break
		self.filename.set(file)

	def clear_file(self, filename):
		if os.path.exists(filename):
			os.remove(filename)

	def _quit(self):
		self.quit()
		self.destroy()

if __name__ == "__main__":
	app = UI()
	app.mainloop()