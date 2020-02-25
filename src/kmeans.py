import numpy as np
import matplotlib.pyplot as plt
import random


def load_data(file):
	data = []
	with open(file, 'r') as f:
		for line in f.readlines():
			line = line.strip().split('\t')
			line = [float(ele) for ele in line]
			data.append(line)
			
		return data

count = 0
def draw_cluster(data, cluster=None, center=None):
	plt.figure()
	if cluster is None:
		x = [line.tolist()[0][0] for line in data]
		y = [line.tolist()[0][1] for line in data]
		plt.scatter(x, y)
	else:
		colors = {'aliceblue':            '#F0F8FF',
					'antiquewhite':         '#FAEBD7',
					'aqua':                 '#00FFFF',
					'aquamarine':           '#7FFFD4',
					'azure':                '#F0FFFF',
					'beige':                '#F5F5DC',
					'bisque':               '#FFE4C4',
					'black':                '#000000',
					'blanchedalmond':       '#FFEBCD',
					'blue':                 '#0000FF',
					'blueviolet':           '#8A2BE2',
					'brown':                '#A52A2A',
					'burlywood':            '#DEB887',
					'cadetblue':            '#5F9EA0',
					'chartreuse':           '#7FFF00',
					'chocolate':            '#D2691E',
					'coral':                '#FF7F50',
					'cornflowerblue':       '#6495ED',
					'cornsilk':             '#FFF8DC',
					'crimson':              '#DC143C',
					'cyan':                 '#00FFFF',
					'darkblue':             '#00008B',
					'darkcyan':             '#008B8B',
					'darkgoldenrod':        '#B8860B',
					'darkgray':             '#A9A9A9',
					'darkgreen':            '#006400',
					'darkkhaki':            '#BDB76B',
					'darkmagenta':          '#8B008B',
					'darkolivegreen':       '#556B2F',
					'darkorange':           '#FF8C00',
					'darkorchid':           '#9932CC',
					'darkred':              '#8B0000',
					'darksalmon':           '#E9967A',
					'darkseagreen':         '#8FBC8F',
					'darkslateblue':        '#483D8B',
					'darkslategray':        '#2F4F4F',
					'darkturquoise':        '#00CED1',
					'darkviolet':           '#9400D3',
					'deeppink':             '#FF1493',
					'deepskyblue':          '#00BFFF',
					'dimgray':              '#696969',
					'dodgerblue':           '#1E90FF',
					'firebrick':            '#B22222',
					'floralwhite':          '#FFFAF0',
					'forestgreen':          '#228B22',
					'fuchsia':              '#FF00FF',
					'gainsboro':            '#DCDCDC',
					'ghostwhite':           '#F8F8FF',
					'gold':                 '#FFD700',
					'goldenrod':            '#DAA520',
					'gray':                 '#808080',
					'green':                '#008000',
					'greenyellow':          '#ADFF2F',
					'honeydew':             '#F0FFF0',
					'hotpink':              '#FF69B4',
					'indianred':            '#CD5C5C',
					'indigo':               '#4B0082',
					'ivory':                '#FFFFF0',
					'khaki':                '#F0E68C',
					'lavender':             '#E6E6FA',
					'lavenderblush':        '#FFF0F5',
					'lawngreen':            '#7CFC00',
					'lemonchiffon':         '#FFFACD',
					'lightblue':            '#ADD8E6',
					'lightcoral':           '#F08080',
					'lightcyan':            '#E0FFFF',
					'lightgoldenrodyellow': '#FAFAD2',
					'lightgreen':           '#90EE90',
					'lightgray':            '#D3D3D3',
					'lightpink':            '#FFB6C1',
					'lightsalmon':          '#FFA07A',
					'lightseagreen':        '#20B2AA',
					'lightskyblue':         '#87CEFA',
					'lightslategray':       '#778899',
					'lightsteelblue':       '#B0C4DE',
					'lightyellow':          '#FFFFE0',
					'lime':                 '#00FF00',
					'limegreen':            '#32CD32',
					'linen':                '#FAF0E6',
					'magenta':              '#FF00FF',
					'maroon':               '#800000',
					'mediumaquamarine':     '#66CDAA',
					'mediumblue':           '#0000CD',
					'mediumorchid':         '#BA55D3',
					'mediumpurple':         '#9370DB',
					'mediumseagreen':       '#3CB371',
					'mediumslateblue':      '#7B68EE',
					'mediumspringgreen':    '#00FA9A',
					'mediumturquoise':      '#48D1CC',
					'mediumvioletred':      '#C71585',
					'midnightblue':         '#191970',
					'mintcream':            '#F5FFFA',
					'mistyrose':            '#FFE4E1',
					'moccasin':             '#FFE4B5',
					'navajowhite':          '#FFDEAD',
					'navy':                 '#000080',
					'oldlace':              '#FDF5E6',
					'olive':                '#808000',
					'olivedrab':            '#6B8E23',
					'orange':               '#FFA500',
					'orangered':            '#FF4500',
					'orchid':               '#DA70D6',
					'palegoldenrod':        '#EEE8AA',
					'palegreen':            '#98FB98',
					'paleturquoise':        '#AFEEEE',
					'palevioletred':        '#DB7093',
					'papayawhip':           '#FFEFD5',
					'peachpuff':            '#FFDAB9',
					'peru':                 '#CD853F',
					'pink':                 '#FFC0CB',
					'plum':                 '#DDA0DD',
					'powderblue':           '#B0E0E6',
					'purple':               '#800080',
					'red':                  '#FF0000',
					'rosybrown':            '#BC8F8F',
					'royalblue':            '#4169E1',
					'saddlebrown':          '#8B4513',
					'salmon':               '#FA8072',
					'sandybrown':           '#FAA460',
					'seagreen':             '#2E8B57',
					'seashell':             '#FFF5EE',
					'sienna':               '#A0522D',
					'silver':               '#C0C0C0',
					'skyblue':              '#87CEEB',
					'slateblue':            '#6A5ACD',
					'slategray':            '#708090',
					'snow':                 '#FFFAFA',
					'springgreen':          '#00FF7F',
					'steelblue':            '#4682B4',
					'tan':                  '#D2B48C',
					'teal':                 '#008080',
					'thistle':              '#D8BFD8',
					'tomato':               '#FF6347',
					'turquoise':            '#40E0D0',
					'violet':               '#EE82EE',
					'wheat':                '#F5DEB3',
					'white':                '#FFFFFF',
					'whitesmoke':           '#F5F5F5',
					'yellow':               '#FFFF00',
					'yellowgreen':          '#9ACD32'}
		colors = list(color.keys())
		for i in range(data.shape[0]):
			color = colors[cluster[i].tolist()[0][0]]
			ele = data[i].tolist()[0]
			plt.scatter(ele[0], ele[1], color=color)

		markers = {0:'8', 1:'*', 2:'h', 3:'p'}
		for i in range(center.shape[0]):
			color = colors[i]
			ele = center[i].tolist()[0]
			plt.scatter(ele[0], ele[1], color=color, linewidths = 7, marker=markers[i])

	global count
	plt.savefig('img/kmeans%d.png' % count)
	count += 1
	plt.close()
	

def draw_bicluster(data, cluster=None, center=None):
	plt.figure()
	if cluster is None:
		x = [line.tolist()[0][0] for line in data]
		y = [line.tolist()[0][1] for line in data]
		plt.scatter(x, y)
	else:
		colors = {'aliceblue':            '#F0F8FF',
					'antiquewhite':         '#FAEBD7',
					'aqua':                 '#00FFFF',
					'aquamarine':           '#7FFFD4',
					'azure':                '#F0FFFF',
					'beige':                '#F5F5DC',
					'bisque':               '#FFE4C4',
					'black':                '#000000',
					'blanchedalmond':       '#FFEBCD',
					'blue':                 '#0000FF',
					'blueviolet':           '#8A2BE2',
					'brown':                '#A52A2A',
					'burlywood':            '#DEB887',
					'cadetblue':            '#5F9EA0',
					'chartreuse':           '#7FFF00',
					'chocolate':            '#D2691E',
					'coral':                '#FF7F50',
					'cornflowerblue':       '#6495ED',
					'cornsilk':             '#FFF8DC',
					'crimson':              '#DC143C',
					'cyan':                 '#00FFFF',
					'darkblue':             '#00008B',
					'darkcyan':             '#008B8B',
					'darkgoldenrod':        '#B8860B',
					'darkgray':             '#A9A9A9',
					'darkgreen':            '#006400',
					'darkkhaki':            '#BDB76B',
					'darkmagenta':          '#8B008B',
					'darkolivegreen':       '#556B2F',
					'darkorange':           '#FF8C00',
					'darkorchid':           '#9932CC',
					'darkred':              '#8B0000',
					'darksalmon':           '#E9967A',
					'darkseagreen':         '#8FBC8F',
					'darkslateblue':        '#483D8B',
					'darkslategray':        '#2F4F4F',
					'darkturquoise':        '#00CED1',
					'darkviolet':           '#9400D3',
					'deeppink':             '#FF1493',
					'deepskyblue':          '#00BFFF',
					'dimgray':              '#696969',
					'dodgerblue':           '#1E90FF',
					'firebrick':            '#B22222',
					'floralwhite':          '#FFFAF0',
					'forestgreen':          '#228B22',
					'fuchsia':              '#FF00FF',
					'gainsboro':            '#DCDCDC',
					'ghostwhite':           '#F8F8FF',
					'gold':                 '#FFD700',
					'goldenrod':            '#DAA520',
					'gray':                 '#808080',
					'green':                '#008000',
					'greenyellow':          '#ADFF2F',
					'honeydew':             '#F0FFF0',
					'hotpink':              '#FF69B4',
					'indianred':            '#CD5C5C',
					'indigo':               '#4B0082',
					'ivory':                '#FFFFF0',
					'khaki':                '#F0E68C',
					'lavender':             '#E6E6FA',
					'lavenderblush':        '#FFF0F5',
					'lawngreen':            '#7CFC00',
					'lemonchiffon':         '#FFFACD',
					'lightblue':            '#ADD8E6',
					'lightcoral':           '#F08080',
					'lightcyan':            '#E0FFFF',
					'lightgoldenrodyellow': '#FAFAD2',
					'lightgreen':           '#90EE90',
					'lightgray':            '#D3D3D3',
					'lightpink':            '#FFB6C1',
					'lightsalmon':          '#FFA07A',
					'lightseagreen':        '#20B2AA',
					'lightskyblue':         '#87CEFA',
					'lightslategray':       '#778899',
					'lightsteelblue':       '#B0C4DE',
					'lightyellow':          '#FFFFE0',
					'lime':                 '#00FF00',
					'limegreen':            '#32CD32',
					'linen':                '#FAF0E6',
					'magenta':              '#FF00FF',
					'maroon':               '#800000',
					'mediumaquamarine':     '#66CDAA',
					'mediumblue':           '#0000CD',
					'mediumorchid':         '#BA55D3',
					'mediumpurple':         '#9370DB',
					'mediumseagreen':       '#3CB371',
					'mediumslateblue':      '#7B68EE',
					'mediumspringgreen':    '#00FA9A',
					'mediumturquoise':      '#48D1CC',
					'mediumvioletred':      '#C71585',
					'midnightblue':         '#191970',
					'mintcream':            '#F5FFFA',
					'mistyrose':            '#FFE4E1',
					'moccasin':             '#FFE4B5',
					'navajowhite':          '#FFDEAD',
					'navy':                 '#000080',
					'oldlace':              '#FDF5E6',
					'olive':                '#808000',
					'olivedrab':            '#6B8E23',
					'orange':               '#FFA500',
					'orangered':            '#FF4500',
					'orchid':               '#DA70D6',
					'palegoldenrod':        '#EEE8AA',
					'palegreen':            '#98FB98',
					'paleturquoise':        '#AFEEEE',
					'palevioletred':        '#DB7093',
					'papayawhip':           '#FFEFD5',
					'peachpuff':            '#FFDAB9',
					'peru':                 '#CD853F',
					'pink':                 '#FFC0CB',
					'plum':                 '#DDA0DD',
					'powderblue':           '#B0E0E6',
					'purple':               '#800080',
					'red':                  '#FF0000',
					'rosybrown':            '#BC8F8F',
					'royalblue':            '#4169E1',
					'saddlebrown':          '#8B4513',
					'salmon':               '#FA8072',
					'sandybrown':           '#FAA460',
					'seagreen':             '#2E8B57',
					'seashell':             '#FFF5EE',
					'sienna':               '#A0522D',
					'silver':               '#C0C0C0',
					'skyblue':              '#87CEEB',
					'slateblue':            '#6A5ACD',
					'slategray':            '#708090',
					'snow':                 '#FFFAFA',
					'springgreen':          '#00FF7F',
					'steelblue':            '#4682B4',
					'tan':                  '#D2B48C',
					'teal':                 '#008080',
					'thistle':              '#D8BFD8',
					'tomato':               '#FF6347',
					'turquoise':            '#40E0D0',
					'violet':               '#EE82EE',
					'wheat':                '#F5DEB3',
					'white':                '#FFFFFF',
					'whitesmoke':           '#F5F5F5',
					'yellow':               '#FFFF00',
					'yellowgreen':          '#9ACD32'}
		colors = list(colors.keys())
		for i in range(data.shape[0]):
			color = colors[int(cluster[i].tolist()[0][0])]
			ele = data[i].tolist()[0]
			plt.scatter(ele[0], ele[1], color=color)

		markers = {0:'8', 1:'*', 2:'h', 3:'p'}
		for i in range(len(center)):
			color = colors[i]
			ele = center[i]
			plt.scatter(ele[0], ele[1], color=color, linewidths = 7, marker=markers[i])

	global count
	plt.savefig('img/bikmeans%d.png' % count)
	count += 1
	plt.close()
	
	
def dist_eclud(rst, snd):
	return np.sqrt(np.sum(np.power(rst - snd, 2)))
	
def rand_center(data, k):
	n = np.shape(data)[1]
	center = np.mat(np.zeros((k, n)))
	for i in range(n):
		min_id = np.min(data[:,i])
		range_id = float(np.max(data[:,i]) - min_id)
		center[:,i] = min_id + range_id * np.random.rand(k, 1)
	
	return center
	
def k_means(data, k, epoch=10):
	m = np.shape(data)[0]
	cluster = np.mat(np.zeros((m, 2)))
	center = rand_center(data, k)
	cluste_changed = True
	while cluste_changed and epoch > 0:
		cluste_changed = False
		for i in range(m):
			min_dist = np.inf
			min_id = -1
			for j in range(k):
				dist = dist_eclud(center[j,:], data[i,:])
				if dist < min_dist:
					min_dist = dist
					min_id = j
				
				if cluster[i,0] != min_id:
					cluste_changed = True
				cluster[i,:] = min_id, min_dist ** 2
			
		for i in range(k):
			pts = data[np.nonzero(cluster[:,0].A == i)[0]]
			center[i,:] = np.mean(pts, axis=0)
		
		#draw_cluster(data, cluster, center)
		epoch -= 1
	return center, cluster
	
def bin_k_means(data, k):
	'''
	@brief	自顶向下
	'''
	m = np.shape(data)[0]
	cluster = np.mat(np.zeros((m, 2)))
	center = np.mean(data, axis=0).tolist()[0]
	center_list = [center]
	for i in range(m):
		cluster[i, 1] = dist_eclud(np.mat(center), data[i,:]) ** 2
		
	best_center_id = None
	best_center = None
	best_cluster = None
	while k > len(center_list):
		lowest_sse = np.inf
		for i in range(len(center_list)):
			pts = data[np.nonzero(cluster[:,0].A == i)[0]]
			cur_center, cur_cluster = k_means(pts, 2)
			sse_split = np.sum(cur_cluster[:,1])
			sse_nosplit = np.sum(cluster[np.nonzero(cluster[:,0].A == i)[0], 1])
			print('split sse is ', sse_split, 'no split sse is ', sse_nosplit)
			if sse_split + sse_nosplit < lowest_sse:
				best_center_id = i
				best_center = cur_center
				best_cluster = cur_cluster.copy()
				lowest_sse = sse_split + sse_nosplit
			
		best_cluster[np.nonzero(best_cluster[:,0].A == 1)[0], 0] = len(center_list)
		best_cluster[np.nonzero(best_cluster[:,0].A == 0)[0], 0] = best_center_id
		center_list[best_center_id] = best_center[0,:].tolist()[0]
		center_list.append(best_center[1:].tolist()[0])
		cluster[np.nonzero(cluster[:,0].A == best_center_id)[0],:]= best_cluster
		
		draw_bicluster(data, cluster, center_list)
		
	return np.mat(center_list), cluster

		
def k_means_test():
	file = 'G:/altas/meching-learning/dataset/testSet.txt'
	data = load_data(file)
	data = np.mat(data)
	draw_cluster(data)
	#k_means(data, 4)
	bin_k_means(data, 4)