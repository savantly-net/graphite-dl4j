package net.savantly.learning.graphite.domain;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.primitives.Ints;

public class GraphiteCsv {
	
	private static final Logger log = LoggerFactory.getLogger(GraphiteCsv.class);
	
	final private List<GraphiteRow> rows = new ArrayList<>();
	final private Map<String, List<GraphiteRow>> rowsGroupedByTarget;
	
	public static GraphiteCsv from(String string) {
		return new GraphiteCsv(string);
	}
	
	public static GraphiteCsv from(File file) throws IOException {
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
		String[] lines = csv.split("\n");
		for (String string : lines) {
			String[] values = string.split(",");
			try {
				this.rows.add(new GraphiteRow(values[0], values[2], values[1]));
			} catch (Exception e) {
				log.warn("failed to import row: ", e.getMessage());
			}
		}
		this.rowsGroupedByTarget = this.rows.stream().collect(Collectors.groupingBy(GraphiteRow::getTarget));
	}

	public List<GraphiteRow> getRows() {
		return rows;
	}
	
	public int getDim0() {
		return this.rowsGroupedByTarget.size();
	}
	public int getDim1() {
		return this.rowsGroupedByTarget.values().stream().map(g->{
			return g.size();
		}).max(Ints::compare).get();
	}
	public int getDim2() {
		return 1;
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
	
	// the epoch of each row becomes the feature
	// the value becomes the label.
	// each unique 'target' is a layer in 3d INDArray
	public DataSet asDataSet3d() {
		int[] shape = new int[] {this.getDim0(), this.getDim1(), this.getDim2()};
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

}
