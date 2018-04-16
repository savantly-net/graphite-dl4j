package net.savantly.learning.graphite.domain;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;
import java.util.stream.Collectors;

import org.datavec.api.util.ndarray.RecordConverter;
import org.datavec.api.writable.Writable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.primitives.Ints;

public class GraphiteCsv {
	
	private static final Logger log = LoggerFactory.getLogger(GraphiteCsv.class);
	
	final private List<GraphiteRow> rows = new ArrayList<>();
	final private Map<String, List<GraphiteRow>> rowsGroupedByTarget;
	
	public static GraphiteCsv from(String string, Function<String[], Boolean> filter) {
		return new GraphiteCsv(string, filter);
	}
	
	public static GraphiteCsv from(String string) {
		return new GraphiteCsv(string);
	}
	
	public static GraphiteCsv from(File file) throws IOException {
		return from(file, null);
	}
	
	public static GraphiteCsv from(File file, Function<String[], Boolean> filter) throws IOException {
		StringBuilder sb = new StringBuilder();
		BufferedReader reader = new BufferedReader(new FileReader(file));
		reader.lines().forEach(s -> {
			sb.append(s);
			sb.append("\n");
		});
		reader.close();
		return from(sb.toString());
	}
	
	public GraphiteCsv(String csv) {
		this(csv, null);
	}
	
	public GraphiteCsv(String csv, Function<String[], Boolean> filter) {
		String[] lines = splitLine(string);
		for (String string : lines) {
			String[] values = string.split(",");
			if(values.length == 3) {
				try {
					boolean allow = true; 
					if (filter != null) {
						allow = filter.apply(values);
					}
					if (allow) {
						this.rows.add(new GraphiteRow(values[0], values[2], values[1]));
					}
				} catch (Exception e) {
					log.warn("failed to import row: ", e.getMessage());
				}
			}
		}
		this.rowsGroupedByTarget = this.rows.stream().collect(Collectors.groupingBy(GraphiteRow::getTarget));
	}

    private static String[] splitLine(final String string) {
        final List<StringBuilder> rv = new ArrayList<>();
        rv.add(new StringBuilder());
        final AtomicBoolean inquotes = new AtomicBoolean(false);
        final AtomicBoolean escaping = new AtomicBoolean(false);

        string.codePoints().forEach(c -> {
            final StringBuilder sb = rv.get(rv.size() - 1);
            if (c == '\\') {
                if (!escaping.get()) {
                    escaping.set(true);
                    return;
                }
            } else if (!escaping.get() && c == '"') {
                inquotes.set(!inquotes.get());
            } else if (!inquotes.get() && c == ',') {
                rv.add(new StringBuilder());
                return;
            }
            escaping.set(false);
            sb.appendCodePoint(c);
        });

        return rv.stream().collect(ArrayList<String>::new, (ArrayList<String> l, StringBuilder e) -> l.add(e.toString()), (a1, a2) -> {
            a1.addAll(a2);
        }).toArray(new String[0]);
    }

	public List<GraphiteRow> getRows() {
		return rows;
	}
	
	public int getSeriesCount() {
		return this.rowsGroupedByTarget.size();
	}
	public int getLongestTimeSteps() {
		return this.rowsGroupedByTarget.values().stream().map(g->{
			return g.size();
		}).max(Ints::compare).get();
	}
	
	// Values only
	// each unique 'target' is an instance of INDArray
	// Each INDArray is shaped [1,$timesteps]
	public List<INDArray> asINDArray() {
		List<INDArray> ndArrays = new ArrayList<>();

		this.rowsGroupedByTarget.values().stream().forEach(g -> {
			int timeSteps = g.size();
			List<Float> values = g.stream().map(r -> {
				return r.getValue();
			}).collect(Collectors.toList());
			
			float[][] data = new float[1][timeSteps];
			for (int i=0; i<values.size(); i++) {
				data[0][i] = values.get(i);
			}
			INDArray ndArray = Nd4j.create(data, 'c');
			ndArrays.add(ndArray);
		});
		
		return ndArrays;
	}
	
	// Values only
	// each unique 'target' is an instance of INDArray
	// Each INDArray is shaped [1,$timesteps]
	public List<INDArray> asINDArray3d() {
		List<INDArray> ndArrays = new ArrayList<>();

		this.rowsGroupedByTarget.values().stream().forEach(g -> {
			int timeSteps = g.size();
			List<Float> values = g.stream().map(r -> {
				return r.getValue();
			}).collect(Collectors.toList());

			INDArray ndArray = Nd4j.create(new int[]{1,1,timeSteps}, 'c');
			ndArrays.add(ndArray);

			for (int i=0; i<values.size(); i++) {
				ndArray.putScalar(0,0,i, values.get(i));
			}
		});
		
		return ndArrays;
	}
	
	// the epoch of each row becomes the feature
	// the value becomes the label.
	// each unique 'target' is a layer in 3d INDArray
	public DataSet asDataSet3d() {
		int[] shape = new int[] {this.getSeriesCount(), this.getLongestTimeSteps(), 1};
		INDArray features = Nd4j.create(shape);
		INDArray labels = Nd4j.create(shape);
		
		AtomicInteger dim0Counter = new AtomicInteger(0);
		this.rowsGroupedByTarget.values().stream().forEach(g -> {
			AtomicInteger dim1Counter = new AtomicInteger(0);
			g.stream().sorted().forEach(r -> {
				features.putScalar(dim0Counter.get(), dim1Counter.get(), 0, r.getEpoch().getMillis());
				labels.putScalar(dim0Counter.get(), dim1Counter.get(), 0, r.getValue());
				dim1Counter.getAndIncrement();
			});
			dim0Counter.incrementAndGet();
		});
		
		return new DataSet(features, labels);
	}
	
	// the epoch of each row becomes the feature
	// the value becomes the label.
	// each unique 'target' is stored in the zeroth dimension
	// shape = [targetCount, 1, timestepcount]
	public DataSet as3dSequence() {
		int[] shape = new int[] {this.getSeriesCount(), 1, this.getLongestTimeSteps()};
		INDArray features = Nd4j.create(shape);
		INDArray labels = Nd4j.create(shape);
		
		AtomicInteger sequenceGroupCounter = new AtomicInteger(0);
		this.rowsGroupedByTarget.values().stream().forEach(g -> {
			AtomicInteger timeStepCounter = new AtomicInteger(0);
			g.stream().sorted().forEach(r -> {
				features.putScalar(sequenceGroupCounter.get(), 0, timeStepCounter.get(), r.getEpoch().getMillis());
				labels.putScalar(sequenceGroupCounter.get(), 0, timeStepCounter.get(), r.getValue());
				timeStepCounter.getAndIncrement();
			});
			sequenceGroupCounter.incrementAndGet();
		});
		
		return new DataSet(features, labels);
	}
	
	public List<List<List<Writable>>> asRecords(int lagSize) {
		return asRecords(lagSize, true);
	}
	
	// lagSize is how many values to use as features [lag window style]
	// default is 1
	public List<List<List<Writable>>> asRecords(int lagSize, boolean includeLabels) {
		List<List<List<Writable>>> results = new ArrayList<>();
		for (DataSet ds : this.asDataSetLagWindow(lagSize)) {
			if (!includeLabels) {
				results.add(RecordConverter.toRecords(ds.getFeatureMatrix()));
			} else {
				results.add(RecordConverter.toRecords(ds));
			}
		}
		return results;
	}
	
	// the epoch of each row becomes the feature
	// the value becomes the label.
	// each unique 'target' is a DataSet in the list
	public List<DataSet> asDataSetList() {
		List<DataSet> dsList = new ArrayList<>();
		this.rowsGroupedByTarget.values().stream().forEach(g -> {
			List<Pair<Long, Float>> values = g.stream().map(r -> {
				return Pair.of(r.getEpoch().getMillis(), r.getValue());
			}).collect(Collectors.toList());
			

			INDArray features = Nd4j.create(new int[] {this.getLongestTimeSteps(), 1}, 'c');
			INDArray labels = Nd4j.create(new int[] {this.getLongestTimeSteps(), 1}, 'c');
			INDArray mask = Nd4j.zeros(new int[] {this.getLongestTimeSteps(), 1}, 'c');
			
			for (int i=0; i<values.size(); i++) {
				features.put(i, 0, values.get(i).getFirst());
				labels.put(i, 0, values.get(i).getSecond());
				mask.put(i, 0, 1);
			}
			DataSet ds = new DataSet(features, labels, mask, mask);
			dsList.add(ds);
		});
		
		return dsList;
	}
	
	public List<DataSet> asLabeledDataSet(int label) {
		List<DataSet> dsList = new ArrayList<>();
		this.rowsGroupedByTarget.values().stream().forEach(g -> {
			List<Pair<Long, Float>> values = g.stream().map(r -> {
				return Pair.of(r.getEpoch().getMillis(), r.getValue());
			}).collect(Collectors.toList());
			

			INDArray features = Nd4j.create(new int[] {this.getLongestTimeSteps(), 1}, 'c');
			INDArray labels = Nd4j.create(new int[] {this.getLongestTimeSteps(), 1}, 'c');
			INDArray mask = Nd4j.zeros(new int[] {this.getLongestTimeSteps(), 1}, 'c');
			
			for (int i=0; i<values.size(); i++) {
				features.put(i, 0, values.get(i).getFirst());
				labels.put(i, 0, values.get(i).getSecond());
				mask.put(i, 0, 1);
			}
			DataSet ds = new DataSet(features, labels, mask, mask);
			dsList.add(ds);
		});
		
		return dsList;
	}
	
	// the previous values become the features for the current value
	// the current value becomes the label
	// each unique 'target' is a DataSet in the list
	public List<DataSet> asDataSetLagWindow(int windowSize) {
		int skipSize = (windowSize*2);
		int timeSteps = this.getLongestTimeSteps() - (skipSize) - 1;
		log.info("windowSize: {}, skipSize: {}, timeSteps: {}", windowSize, skipSize, timeSteps);
		if (timeSteps < 1) {
			throw new RuntimeException("there are not enough records to create a lag");
		}
		List<DataSet> dsList = new ArrayList<>();
		this.rowsGroupedByTarget.values().stream().forEach(g -> {
			List<Pair<Long, Float>> values = g.stream().map(r -> {
				return Pair.of(r.getEpoch().getMillis(), r.getValue());
			}).filter(r -> r.getValue() != null).collect(Collectors.toList());
			
			log.info("values.size(): {}", values.size());
			

			INDArray features = Nd4j.create(new int[] {timeSteps, windowSize}, 'c');
			INDArray labels = Nd4j.create(new int[] {timeSteps, 1}, 'c');
			INDArray mask = Nd4j.zeros(new int[] {timeSteps, 1}, 'c');
			
			LinkedList<Float> queue = new LinkedList<>();
			for (int i=0; i<=(values.size()-windowSize); i++) {
				float value = values.get(i).getValue();
				queue.push(value);
				if((queue.size() > windowSize) && (i<timeSteps+windowSize)) {
					for(int j=windowSize; j>0; j--) {
						try {
						features.put(i-windowSize, j-1, queue.get(j));
						} catch (Exception e) {
							log.error("{}", e);
						}
					}
					queue.removeLast();
					labels.put(i-windowSize, 0, value);
					mask.put(i-windowSize, 0, 1);
				}
			}
			DataSet ds = new DataSet(features, labels);
			dsList.add(ds);
		});
		
		return dsList;
	}
	
    public INDArray as3dMatrix(int lagSize, boolean includeLabels) {
    	List<List<List<Writable>>> sequenceList = this.asRecords(lagSize, includeLabels);
    	
    	int sequenceSize = 1;//sequenceList.size();
    	
    	INDArray matrix2d = RecordConverter.toMatrix(sequenceList.get(0));
    	
    	int featureSize = matrix2d.size(1);
    	int timeStepSize = matrix2d.size(0);
    	INDArray features = Nd4j.create(new int[] {sequenceSize, featureSize, timeStepSize});
    	
    	for (int i = 0; i < sequenceList.size(); i++) {
			for (int j = 0; j < featureSize; j++) {
				for (int k = 0; k < timeStepSize; k++) {
					features.putScalar(i, j, k, matrix2d.getFloat(k, j));
				}
			}
		}
        return features;
    }

	public List<DataSet> asLabeledDataSet(Boolean label) {
		return this.asLabeledDataSet(label?1:0);
	}


}
