package net.savantly.learning.graphite.convert;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import net.savantly.learning.graphite.domain.GraphiteMultiSeries;

public class GraphiteToDataSet {
	
	public static DataSet toDataSet(double label, GraphiteMultiSeries... multiSeries) {
		
		List<String> targets = Arrays.stream(multiSeries).flatMap(s -> {
			return s.stream().map(g -> {
				return g.getTarget();
			});
		}).sorted().collect(Collectors.toList());
		
		long dataPointCount = Arrays.stream(multiSeries).flatMap(s -> {
			return s.stream().flatMap(g -> {
				return g.getDatapoints().stream().map(d -> {
					return d.size();
				});
			});
		}).count();
		

		INDArray features = Nd4j.create((int)dataPointCount, 3);
		INDArray labels = Nd4j.create(1, 1);
		labels.assign(label);
		
		AtomicInteger exampleRow = new AtomicInteger(0);
		Arrays.stream(multiSeries).forEach(s -> {
			s.stream().forEach(g -> {
				g.getDatapoints().stream().forEach(d -> {
					if (d.getValue() != null) {
						INDArray row = features.getRow(exampleRow.getAndIncrement());
						row.getColumn(0).addi(targets.indexOf(g.getTarget()));
						row.getColumn(1).addi(d.getValue());
						row.getColumn(2).addi(d.getEpoc());
					}
				});
			});
		});
		
		DataSet ds = new DataSet(features, labels);
		
		return ds;
	}

}
