package net.savantly.learning.graphite.learners.timeseries;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

import org.datavec.api.records.reader.SequenceRecordReader;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator.AlignmentMode;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.primitives.Ints;

import net.savantly.learning.graphite.convert.GraphiteToDataSet;
import net.savantly.learning.graphite.domain.GraphiteMultiSeries;
import net.savantly.learning.graphite.learners.MultiLayerLearnerBase;
import net.savantly.learning.graphite.sequence.GraphiteSequenceRecordReader;

public class GraphiteSeriesClassifier extends MultiLayerLearnerBase {
	private static final Logger log = LoggerFactory.getLogger(GraphiteSeriesClassifier.class);

	private List<GraphiteMultiSeries> positiveExamples = new ArrayList<>();
	private List<GraphiteMultiSeries> negativeExamples = new ArrayList<>();
	private File workingDirectory;
	private int miniBatchSize = 1;
	private boolean doRegression = false;
	private double percentTrain = 0.75;
	private List<Pair<INDArray, INDArray>> trainingData;
	private List<Pair<INDArray, INDArray>> testingData;
	private List<Pair<INDArray, INDArray>> allData;
	private double learningRate = 0.007;
	private List<IterationListener> iterationListeners;
	private int numberOfIterations;
	private DataNormalization normalizer;
	private int featureCount;

	private GraphiteSeriesClassifier() { }

	public static GraphiteSeriesClassifier builder() {
		return new GraphiteSeriesClassifier();
	}

	public GraphiteSeriesClassifier build() throws IOException {
		if (this.workingDirectory == null) {
			this.workingDirectory = Files.createDirectories(Paths.get("data")).toFile();
		}
		
		Pair<Integer, GraphiteMultiSeries>[] multiSeries = 
				new Pair[this.negativeExamples.size() + this.positiveExamples.size()];
		
		final AtomicInteger pairCounter = new AtomicInteger(0);
		this.negativeExamples.forEach(e -> {
			multiSeries[pairCounter.getAndIncrement()] = Pair.of(0, e);
		});
		this.positiveExamples.forEach(e -> {
			multiSeries[pairCounter.getAndIncrement()] = Pair.of(1, e);
		});
		
		this.allData = GraphiteToDataSet.toTimeSeriesNDArrayPairs(multiSeries);
		
		int trainCount = (int) Math.round(percentTrain * this.allData.size()); 
		
		if(trainCount == this.allData.size() && trainCount > 1) {
			trainCount -= 1;
		} else if (trainCount == 1) {
			log.warn("Not enough data to do testing");
		}
		
		this.trainingData = this.allData.subList(0, trainCount);
		this.testingData = this.allData.subList(trainCount, this.allData.size());
		
		this.featureCount = this.allData.stream().max((p1,p2) -> {
			return Ints.compare(p1.getFirst().length(), p2.getFirst().length());
		}).get().getFirst().length();
		
		return this;
	}
	
	// Pads the beginning with the avg value of the row
	public INDArray reshapeNDArray(INDArray ndArray) {
		int timeSteps = ndArray.size(1);
		INDArray replacement = Nd4j.create(1, this.featureCount);
		double avgValue = ndArray.mean(1).getDouble(0);
		int missingStepCount = this.featureCount - timeSteps;
		// Pad the beginning of the replacement array with the avg value of the original array
		for(int i = 0; i < missingStepCount; i++) {
			replacement.putScalar(i, avgValue);
		}
		for(int i = missingStepCount; i < this.featureCount; i++) {
			replacement.putScalar(i, ndArray.getDouble(i-missingStepCount));
		}
		return replacement;
	}
	
	public DataSetIterator createDataSetIterator(List<Pair<INDArray, INDArray>> dataPairs) {
		dataPairs.forEach(p->{
			int timeSteps = p.getLeft().size(1);
			if(timeSteps != this.featureCount) {
				if(timeSteps < this.featureCount) {
					p.setFirst(this.reshapeNDArray(p.getLeft()));
				}
			}
		});
		SequenceRecordReader featuresReader = new GraphiteSequenceRecordReader(dataPairs.stream().map(p -> {
			return p.getFirst();
		}).collect(Collectors.toList()));
		
		SequenceRecordReader labelsReader = new GraphiteSequenceRecordReader(dataPairs.stream().map(p -> {
			return p.getSecond();
		}).collect(Collectors.toList()));
		
		SequenceRecordReaderDataSetIterator iterator = 
				new SequenceRecordReaderDataSetIterator(
						featuresReader, labelsReader, miniBatchSize, this.getNumberOfPossibleLabels(), this.doRegression, AlignmentMode.ALIGN_END);
		return iterator;
	}

	public DataSetIterator getTestingDataSets() {
		return createDataSetIterator(this.testingData);
	}

	public DataSetIterator getTrainingDataSets() {
		return createDataSetIterator(this.trainingData);
	}

	public List<GraphiteMultiSeries> getPostiveExamples() {
		return positiveExamples;
	}

	public GraphiteSeriesClassifier setPositiveExamples(List<GraphiteMultiSeries> positiveTrainingExamples) {
		this.positiveExamples = positiveTrainingExamples;
		return this;
	}

	public List<GraphiteMultiSeries> getNegativeExamples() {
		return negativeExamples;
	}

	public GraphiteSeriesClassifier setNegativeExamples(List<GraphiteMultiSeries> negativeTrainingExamples) {
		this.negativeExamples = negativeTrainingExamples;
		return this;
	}

	public File getWorkingDirectory() {
		return workingDirectory;
	}

	public GraphiteSeriesClassifier setWorkingDirectory(File workingDirectory) {
		this.workingDirectory = workingDirectory;
		return this;
	}

	public int getMiniBatchSize() {
		return miniBatchSize;
	}

	public GraphiteSeriesClassifier setMiniBatchSize(int miniBatchSize) {
		this.miniBatchSize = miniBatchSize;
		return this;
	}

	public boolean isDoRegression() {
		return doRegression;
	}

	public GraphiteSeriesClassifier setDoRegression(boolean doRegression) {
		this.doRegression = doRegression;
		return this;
	}

	public int getNumberOfIterations() {
		return numberOfIterations;
	}

	public GraphiteSeriesClassifier setNumberOfIterations(int numberOfIterations) {
		this.numberOfIterations = numberOfIterations;
		return this;
	}

	public DataNormalization getNormalizer() {
		return this.normalizer;
	}

	public GraphiteSeriesClassifier setNormalizer(DataNormalization normalizer) {
		this.normalizer = normalizer;
		return this;
	}

	public List<IterationListener> getIterationListeners() {
		return iterationListeners;
	}

	public GraphiteSeriesClassifier setIterationListeners(List<IterationListener> iterationListeners) {
		this.iterationListeners = iterationListeners;
		return this;
	}

	public GraphiteSeriesClassifier addIterationListener(IterationListener listener) {
		this.iterationListeners.add(listener);
		return this;
	}


	public double getLearningRate() {
		return this.learningRate;
	}

	public GraphiteSeriesClassifier setLearningRate(double learningRate) {
		this.learningRate = learningRate;
		return this;
	}

	@Override
	public int getNumberOfPossibleLabels() {
		if (this.doRegression) {
			return this.featureCount;
		} else return 2;
	}

	@Override
	public int getFeatureCount() {
		return this.featureCount;
	}

}
