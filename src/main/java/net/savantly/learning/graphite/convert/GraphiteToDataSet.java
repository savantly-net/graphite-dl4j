package net.savantly.learning.graphite.convert;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import com.google.common.primitives.Ints;

import net.savantly.learning.graphite.domain.GraphiteDatapoint;
import net.savantly.learning.graphite.domain.GraphiteMultiSeries;
import net.savantly.learning.graphite.domain.GraphiteSeries;

public class GraphiteToDataSet {

	// Good for small number of features
	public static DataSet toDataSet(double label, GraphiteMultiSeries... multiSeries) {

		List<String> targets = Arrays.stream(multiSeries).flatMap(s -> {
			return s.stream().map(g -> {
				return g.getTarget();
			});
		}).distinct().sorted().collect(Collectors.toList());

		long dataPointCount = Arrays.stream(multiSeries).flatMap(s -> {
			return s.stream().map(g -> {
				return g.getDatapoints().stream().count();
			});
		}).collect(Collectors.summingLong(s->{
			return s;
		}));

		float[][] featuresArray = new float[(int) dataPointCount][3];

		AtomicInteger exampleRow = new AtomicInteger(-1);
		Arrays.stream(multiSeries).forEach(s -> {
			s.stream().forEach(g -> {
				g.getDatapoints().stream().forEach(d -> {
					if (d.getValue() != null) {
						float[] row = featuresArray[exampleRow.incrementAndGet()];
						row[0] = (float) targets.indexOf(g.getTarget());
						row[1] = d.getValue().floatValue();
						row[2] = d.getEpoc().floatValue();
					}
				});
			});
		});

		INDArray features = Nd4j.create(featuresArray);
		INDArray labels = Nd4j.create(1, 1);
		labels.assign(label);

		DataSet ds = new DataSet(features, labels);

		return ds;
	}

	// the targets from each multi-series are extracted, grouped and sorted by
	// epoch.
	// so that we have a list of dataset. One per unique target
	public static List<DataSet> toTimeSeriesDataSetList(Pair<Integer, GraphiteMultiSeries>[] multiSeriesPairs) {

		Stream<GraphiteSeries> targetGroupsStream = Arrays.stream(multiSeriesPairs).flatMap(s -> {
			return s.getValue().stream().map(g -> {
				g.getDatapoints().forEach(d -> {
					d.setLabel(s.getKey().toString());
				});
				return g;
			});
		});

		Map<String, List<GraphiteSeries>> targetGroups = targetGroupsStream
				.collect(Collectors.groupingBy(GraphiteSeries::getTarget));

		final List<DataSet> dataSets = new ArrayList<DataSet>();

		Set<String> targets = targetGroups.keySet();
		for (String key : targets) {
			List<GraphiteSeries> series = targetGroups.get(key);
			List<GraphiteDatapoint> datapointList = series.stream().flatMap(d -> {
				return d.getDatapoints().stream();
			}).sorted().filter(d -> d.getValue() != null && !Double.isNaN(d.getValue().doubleValue())).collect(Collectors.toList());

			final int dataPointCount = datapointList.size();

			final float[] values = new float[dataPointCount];
			final AtomicInteger valueCounter = new AtomicInteger(0);
			datapointList.forEach(d -> {
				values[valueCounter.getAndIncrement()] = d.getValue().floatValue();
			});

			final float[] labelArray = new float[dataPointCount];
			final AtomicInteger labelCounter = new AtomicInteger(0);
			datapointList.forEach(d -> {
				labelArray[labelCounter.getAndIncrement()] = Float.parseFloat(d.getLabel());
			});

			final INDArray ndValues = Nd4j.create(Nd4j.createBuffer(values));
			final INDArray ndLabels = Nd4j.create(Nd4j.createBuffer(labelArray));
			dataSets.add(new DataSet(ndValues, ndLabels));
		}

		return dataSets;
	}
	
	// the targets from each multi-series are extracted, grouped and sorted by
		// epoch.
		// so that we have a list of dataset. One per unique target
		public static List<DataSet> toTimeSeriesDataSetList(GraphiteMultiSeries... multiSeries) {

			Map<String, List<GraphiteSeries>> targetGroups = Arrays.stream(multiSeries).flatMap(s -> {
				return s.stream().map(g -> {
					return g;
				});
			}).collect(Collectors.groupingBy(GraphiteSeries::getTarget));

			final List<DataSet> dataSets = new ArrayList<DataSet>();

			Set<String> targets = targetGroups.keySet();
			for (String key : targets) {
				List<GraphiteSeries> series = targetGroups.get(key);
				List<GraphiteDatapoint> datapointList = series.stream().flatMap(d -> {
					return d.getDatapoints().stream();
				}).sorted().filter(d -> d.getValue() != null && !Double.isNaN(d.getValue().doubleValue())).collect(Collectors.toList());

				final int dataPointCount = datapointList.size();

				final float[] values = new float[dataPointCount];
				final AtomicInteger valueCounter = new AtomicInteger(0);
				datapointList.forEach(d -> {
					values[valueCounter.getAndIncrement()] = d.getValue().floatValue();
				});

				final INDArray ndValues = Nd4j.create(Nd4j.createBuffer(values));
				dataSets.add(new DataSet(ndValues, Nd4j.create(new float[] {0f})));
			}

			return dataSets;
		}
	
	// the targets from each multi-series are extracted, grouped and sorted by
	// epoch.
	// so that we have a list of INDArray Pairs. One pair per unique target <features, labels>
	public static List<Pair<INDArray, INDArray>> toTimeSeriesNDArray(Pair<Integer, GraphiteMultiSeries>[] multiSeriesPairs) {
		List<Pair<INDArray, INDArray>> results = new ArrayList<Pair<INDArray, INDArray>>();
		// just using the value of the datapoint as an input for each example
		// and the provided label in the 2nd dimension
		final int featureSize = 2;

		Map<String, List<GraphiteSeries>> targetGroups = Arrays.stream(multiSeriesPairs).flatMap(s -> {
			return s.getValue().stream().map(g -> {
				g.getDatapoints().forEach(d -> {
					d.setLabel(s.getKey().toString());
				});
				return g;
			});
		}).collect(Collectors.groupingBy(GraphiteSeries::getTarget));
		
/*		final int timeSeriesLength = targetGroups.values().stream().map(s -> {
			return s.stream().max((s1, s2) -> {
				return Ints.compare(s1.getDatapoints().size(), s2.getDatapoints().size());
			}).get();
		}).findFirst().get().getDatapoints().size();*/

		Set<String> targets = targetGroups.keySet();
		for (String key : targets) {
			List<GraphiteSeries> series = targetGroups.get(key);
			List<GraphiteDatapoint> datapointList = series.stream().flatMap(d -> {
				return d.getDatapoints().stream();
			}).sorted().filter(d -> d.getValue() != null && !Double.isNaN(d.getValue().doubleValue())).collect(Collectors.toList());

			final int dataPointCount = datapointList.size();

			final float[] values = new float[dataPointCount];
			final AtomicInteger valueCounter = new AtomicInteger(0);
			datapointList.forEach(d -> {
				values[valueCounter.getAndIncrement()] = d.getValue().floatValue();
			});

			final float[] labelArray = new float[dataPointCount];
			final AtomicInteger labelCounter = new AtomicInteger(0);
			datapointList.forEach(d -> {
				labelArray[labelCounter.getAndIncrement()] = Float.parseFloat(d.getLabel());
			});
			
			if(values.length > 0) {
				final INDArray ndValues = Nd4j.create(Nd4j.createBuffer(values));
				final INDArray ndLabels = Nd4j.create(Nd4j.createBuffer(labelArray));
				results.add(Pair.of(ndValues, ndLabels));
			}
		}

		return results;
	}

}
