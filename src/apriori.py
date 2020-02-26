def create_c1(data):
	c1 = []
	for item in data:
		if [item] not in c1:
			c1.append([item])
		
	c1.sort()
	return map(frozenset, c1)
	
def scan_d(data, c_k, min_support):
	'''
	@brief	计算support
	'''
	ss_cnt = {}
	for item in data:
		for can in c_k:
			if can.issubset(item):
				if not ss_cnt.has_key(can):
					ss_cnt[can] = 1
				else:
					ss_cnt[can] += 1
	
	num_items = len(data)
	ret_list = []
	support_data = {}
	for key in ss_cnt:
		support = ss_cnt[key] / num_items
		if support >= min_support:
			ret_list.insert(0, key)
		
		support_data[key] = support
	
	return ret_list, support_data
	
def aprior_gen(lk, k):
	ret_list = []
	for i in range(len(lk)):
		for j in range(i + 1, len(lk)):
			l1 = list(lk[i])[:k-2]
			l2 = list(lk[j])[:k-2]
			l2.sort()
			l1.sort()
			if l1 == l2:
				ret_list.append(lk[i] | lk[j])
	
	return ret_list
	
def apriori(data, min_support=0.5):
	c1 = create_c1(data)
	d = map(set, data)
	l1, support_data = scan_d(d, c1, min_support)
	l = [l1]
	k = 2
	while len(l[k - 2]) > 0:
		ck = aprior_gen(l[k - 2], k)
		lk, sup_k = scan_d(d, ck, min_support)
		support_data.update(sup_k)
		l.append(lk)
		k += 1
	
	return l, support_data
	
def generate_rules(l, support_data, min_conf=0.7):  #support_data is a dict coming from scanD
    big_rule_list = []
    for i in range(1, len(l)):#only get the sets with two or more items
        for freq_set in l[i]:
            h1 = [frozenset([item]) for item in freq_set]
            if (i > 1):
                rules_from_conseq(freq_set, h1, support_data, big_rule_list, min_conf)
            else:
                calc_conf(freq_set, h1, support_data, big_rule_list, min_conf)
    return big_rule_list         

def calc_conf(freq_set, h, support_data, brl, min_conf=0.7):
    prunedh = [] #create new list to return
    for conseq in h:
        conf = support_data[freq_set]/support_data[freq_set-conseq] #calc confidence
        if conf >= min_conf: 
            print(freq_set-conseq,'-->',conseq,'conf:',conf)
            brl.append((freq_set-conseq, conseq, conf))
            prunedh.append(conseq)
    return prunedh

def rules_from_conseq(freq_set, h, support_data, brl, min_conf=0.7):
    m = len(h[0])
    if (len(freq_set) > (m + 1)): #try further merging
        hmp1 = aprioriGen(h, m+1)#create hm+1 new candidates
        hmp1 = calc_conf(freq_set, hmp1, support_data, brl, min_conf)
        if (len(hmp1) > 1):    #need at least two sets to merge
            rules_from_conseq(freq_set, hmp1, support_data, brl, min_conf)
            
def pntRules(rulelist, item_meaning):
    for rule_tup in rulelist:
        for item in rule_tup[0]:
            print(item_meaning[item])
        print("           -------->")
        for item in rule_tup[1]:
            print(item_meaning[item])
        print("confidence: %f" % rule_tup[2])
        print()
		

def load_dataset():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]
	
def apriori_test():
	data = load_dataset()
	l, support_data = apriori(data)
	rules = generate_rules(l, support_data)
	print(rules)