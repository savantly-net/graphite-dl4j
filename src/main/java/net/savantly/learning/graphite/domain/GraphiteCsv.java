package net.savantly.learning.graphite.domain;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetUtil;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class GraphiteCsv {
	
	private static final Logger log = LoggerFactory.getLogger(GraphiteCsv.class);
	
	private List<GraphiteRow> rows = new ArrayList<>();
	
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
	}

	public List<GraphiteRow> getRows() {
		return rows;
	}

	public void setRows(List<GraphiteRow> rows) {
		this.rows = rows;
	}
	
	public List<INDArray> asINDArray() {
		List<INDArray> ndArrays = new ArrayList<>();
		
		Map<String, List<GraphiteRow>> targetGroups = this.rows.stream().collect(Collectors.groupingBy(GraphiteRow::getTarget));
		targetGroups.values().stream().forEach(g -> {
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
	
	public DataSet asDataSet3d() {
		List<INDArray> arrayList = this.asINDArray();
		//DataSetUtil.
		return null;
	}

}
