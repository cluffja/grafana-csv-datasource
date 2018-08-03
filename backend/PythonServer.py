from flask import Flask, request, jsonify, json, abort
from flask_cors import CORS, cross_origin
from cachetools import cached, TTLCache
from itertools import combinations

import pandas as pd
import glob
import os
import csv
import sys
import getopt
import time
import pickle
import psutil


app = Flask(__name__)

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
methods = ('GET', 'POST')
path = './'
metric_finders = {}
metric_readers = {}
annotation_readers = {}
panel_readers = {}
timesFileOpen = 1
cache = TTLCache(maxsize=10, ttl=300)

#-------------------------------------------------------------------------------
def add_reader(name, reader):
	metric_readers[name] = reader

#-------------------------------------------------------------------------------
def add_finder(name, finder):
	metric_finders[name] = finder

#-------------------------------------------------------------------------------
def add_annotation_reader(name, reader):
	annotation_readers[name] = reader

#-------------------------------------------------------------------------------
def add_panel_reader(name, reader):
	panel_readers[name] = reader

# ------------------------------------------------------------------------------
@app.route('/<folder>/', methods=methods)
@app.route('/<folder>', methods=methods)
@cross_origin()
def hello_world(folder):
	print(path+str(folder))
	if (os.path.isdir(path+str(folder))):
		return 'CSV Python Grafana datasource for '+str(folder)
	else:
		abort(404)

# ------------------------------------------------------------------------------
@app.route('/<folder>/sources', methods=methods)
@cross_origin(max_age=600)
def query_routes(folder):
	results = []
	for file in glob.glob(path+str(folder)+"/*.csv"):
		results.append(os.path.basename(file)[:-4])
	return jsonify(results)

# ------------------------------------------------------------------------------
@app.route('/<folder>/search', methods=methods)
@cross_origin()
def find_metrics(folder):
	global timesFileOpen
	req = request.get_json()
	source = req.get('source', '')
	with open(path+str(folder)+"/"+source+".csv", 'r') as csvfile:
		print("Times opened a file: ", timesFileOpen)
		timesFileOpen+= 1
		dialect = csv.Sniffer().sniff(csvfile.readline())
		csvfile.seek(0)
		reader = csv.reader(csvfile, dialect)
		fieldnames = reader.__next__()
	target = req.get('target', '')
	metrics = []
	for key in fieldnames:
		if key.find(target) != -1:
			metrics.append(key)
	process = psutil.Process(os.getpid())
	print("Memory usage after find_metrics: ", (process.memory_info().rss)/10**6)
	return jsonify(metrics)

# ------------------------------------------------------------------------------
def dataframe_to_response(target, df):
	response = []
	if df.empty:
	    return response
	if isinstance(df, pd.Series):
		response.append(_series_to_response(df, target))
	elif isinstance(df, pd.DataFrame):
		for col in df:
			response.append(_series_to_response(df[col], target))
	else:
	    abort(404, Exception('Received object is not a dataframe or series.'))

	return response

#-------------------------------------------------------------------------------
def dataframe_to_json_table(target, df):
	response = []
	if df.empty:
		return response
	if isinstance(df, pd.DataFrame):
		response.append({'type': 'table',
                         'columns': df.columns.map(lambda col: {"text": col}).tolist(),
                         'rows': df.where(pd.notnull(df), None).values.tolist()})
	else:
		abort(404, Exception('Received object is not a dataframe.'))
	return response

#-------------------------------------------------------------------------------
def annotations_to_response(target, df):
    response = []
    # Single series with DatetimeIndex and values as text
    if isinstance(df, pd.Series):
        for timestamp, value in df.iteritems():
            response.append({
                "annotation": target, # The original annotation sent from Grafana.
                "time": timestamp.value // 10 ** 6, # Time since UNIX Epoch in milliseconds. (required)
                "title": value, # The title for the annotation tooltip. (required)
                #"tags": tags, # Tags for the annotation. (optional)
                #"text": text # Text for the annotation. (optional)
            })
    # Dataframe with annotation text/tags for each entry
    elif isinstance(df, pd.DataFrame):
        for timestamp, row in df.iterrows():
            annotation = {
                "annotation": target,  # The original annotation sent from Grafana.
                "time": timestamp.value // 10 ** 6,  # Time since UNIX Epoch in milliseconds. (required)
                "title": row.get('title', ''),  # The title for the annotation tooltip. (required)
            }
            if 'text' in row:
                annotation['text'] = str(row.get('text'))
            if 'tags' in row:
                annotation['tags'] = str(row.get('tags'))
            response.append(annotation)
    else:
        abort(404, Exception('Received object is not a dataframe or series.'))
    return response

#-------------------------------------------------------------------------------
def _series_to_annotations(df, target):
    if df.empty:
        return {'target': '%s' % (target),
                'datapoints': []}
    sorted_df = df.dropna().sort_index()
    timestamps = (sorted_df.index.astype(pd.np.int64) // 10 ** 6).values.tolist()
    values = sorted_df.values.tolist()
    return {'target': '%s' % (df.name),
            'datapoints': list(zip(values, timestamps))}

#-------------------------------------------------------------------------------
def _series_to_response(df, target):
	if df.empty:
		return {'target': '%s' % (target),
				'datapoints': []}
	sorted_df = df.dropna().sort_index()
	try:
		timestamps = (sorted_df.index.astype(pd.np.int64) // 10 ** 6).values.tolist() # New pandas version
	except:
		timestamps = (sorted_df.index.astype(pd.np.int64) // 10 ** 6).tolist()
	values = sorted_df.values.tolist()
	process = psutil.Process(os.getpid())
	print("Memory usage after series_to_response: ", (process.memory_info().rss)/10**6)
	return {'target': '%s' % (df.name),
            'datapoints': list(zip(values, timestamps))}

#-------------------------------------------------------------------------------
@app.route('/<folder>/query', methods=methods)
@cross_origin(max_age=600)
def query_metrics(folder):
	startGetCSVs = time.time()
	global timesFileOpen
	req = request.get_json()
	results = []
	CSVs = {}
	for target in req['targets']:
		source = target['source']
		if source=="":
			return jsonify(results)
		if source not in CSVs:
			CSVs[source] = get_CSVs(folder, source)
	endGetCSVs = time.time()
	print("Get CSVs time: ", endGetCSVs - startGetCSVs)
	startStrip = time.time()
	for target in req['targets']:
		source = target['source']
		query_results = CSVs[source].filter(items=[target["target"]])
		if (query_results[target["target"]].dtype==object):
			query_results[target["target"]] = pd.to_numeric(query_results[target["target"]].str.replace(',','.'), errors='coerce')
		query_results = query_results[(query_results.index >= pd.Timestamp(req['range']['from']).to_pydatetime()) & (query_results.index <= pd.Timestamp(req['range']['to']).to_pydatetime())]
		if target.get('type', 'timeserie') == 'table':
		    results.extend(dataframe_to_json_table(target, query_results))
		else:
		    results.extend(dataframe_to_response(target, query_results))
	endStrip = time.time()
	print("Query_metrics time: ", endStrip - startStrip)
	process = psutil.Process(os.getpid())
	print("Memory usage after query_metrics: ", (process.memory_info().rss)/10**6)
	return jsonify(results)

#-------------------------------------------------------------------------------
def get_CSVs(folder, source):
	with open(path+str(folder)+"/"+source+".csv", 'r') as csvfile:
		try:
			with open(path+"config.pkl", "r+b") as f:
				parsedFiles = pickle.load(f)
		except IOError:
			parsedFiles = {"fileName":"boolean"}
			with open(path+"config.pkl", "wb") as f:
				pickle.dump(parsedFiles, f)
		dialect = csv.Sniffer().sniff(csvfile.readline())
		csvData = pd.read_csv(path+str(folder)+"/"+source+".csv",index_col=0 , dialect=dialect, encoding='latin1')	
		if source in parsedFiles and parsedFiles[source] == "True":
			print("Skip datetime parsing...")
			csvData.index = pd.to_datetime(csvData.index, format=r"%Y-%m-%d %H:%M:%S").tz_localize("Universal")
			process = psutil.Process(os.getpid())
			print("Memory usage after get_CSVs without parsing: ", (process.memory_info().rss)/10**6)
			return csvData
		else:
			print("Parsing datetimes...")
			csvData.index = pd.to_datetime(csvData.index).tz_localize('Universal')
			csvData.to_csv(path+str(folder)+"/"+source+".csv")
			parsedFiles[source] = "True"
			with open(path+"config.pkl", "wb") as f:
				pickle.dump(parsedFiles, f)
			process = psutil.Process(os.getpid())
			print("Memory usage after get_CSVs with parsing: ", (process.memory_info().rss)/10**6)
			return csvData

#-------------------------------------------------------------------------------
@app.route('/<folder>/annotations', methods=methods)
@cross_origin(max_age=600)
def query_annotations(folder):
	print(request.get_json())
	req = request.get_json()
	results = []
	ts_range = {'$gt': pd.Timestamp(req['range']['from']).to_pydatetime(),
				'$lte': pd.Timestamp(req['range']['to']).to_pydatetime()}
	query = req['annotation']['query']
	if ':' not in query:
		abort(404, Exception('Target must be of type: <finder>:<metric_query>, got instead: ' + query))
	finder, target = query.split(':', 1)
	results.extend(annotations_to_response(query, annotation_readers[finder](target, ts_range)))
	return jsonify(results)

#-------------------------------------------------------------------------------
@app.route('/<folder>/panels', methods=methods)
@cross_origin()
def get_panel(folder):
	print(request.headers, request.get_json())
	req = request.args
	ts_range = {'$gt': pd.Timestamp(int(req['from']), unit='ms').to_pydatetime(),
                '$lte': pd.Timestamp(int(req['to']), unit='ms').to_pydatetime()}
	query = req['query']
	if ':' not in query:
		abort(404, Exception('Target must be of type: <finder>:<metric_query>, got instead: ' + query))
	finder, target = query.split(':', 1)
	return panel_readers[finder](target, ts_range)

#-------------------------------------------------------------------------------
def main(argv):
	global path
	port = 3003
	debug = True
	addr = '0.0.0.0'
	try:
	  opts, args = getopt.getopt(argv,"hvp:f:a:",["port=","folder=","addr="])
	except getopt.GetoptError:
	  print ('PythonServer.py -p <port> -f <folder>')
	  sys.exit(2)
	for opt, arg in opts:
	  if opt == '-h':
	     print('PythonServer.py -p <port> -f <folder>')
	     sys.exit()
	  elif opt in ("-a", "--addr"):
	     addr = arg
	  elif opt in ("-p", "--port"):
	     port = int(arg)
	  elif opt in ("-f", "--folder"):
	     path = arg
	  elif opt in ("-v"):
	     debug = True
	print('debug', debug)
	app.run(host=addr, port=port, debug=debug)

if __name__ == '__main__':
	main(sys.argv[1:])