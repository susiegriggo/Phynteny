"""
Convert pickle to alternate embedding type
""" 

#imports  
from phynteny_utils import statistics 

x = pickle5.load(open('/home/grig0076/scratch/phynteny_data/phynteny_data_test_X.pkl', 'rb'))
x_values = list(x.values())
y = pickle5.load(open('/home/grig0076/scratch/phynteny_data/phynteny_data_test_y.pkl', 'rb'))
y_values = list(y.values())

y_rows = []
x_rows = []
for i in range(len(x_values)):

    idx = statistics.get_masked(x_values[i],10)

    y_rows.append(y_values[i][idx].reshape(1,10))
    x_rows.append(x_values[i][:,:10])

x_rows_dict = dict(zip(list(x.keys()), x_rows))
y_rows_dict = dict(zip(list(y.keys()), y_rows))

pickle5.dump(x_rows_dict, open('/home/grig0076/scratch/phynteny_data/phynteny_data_test_X_row.pkl', 'wb'))
pickle5.dump(y_rows_dict, open('/home/grig0076/scratch/phynteny_data/phynteny_data_test_y_row.pkl', 'wb'))

