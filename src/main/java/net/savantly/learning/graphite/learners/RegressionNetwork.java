package net.savantly.learning.graphite.learners;

import java.util.Collection;
import java.util.LinkedList;

import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionSequenceRecordReader;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class RegressionNetwork {
	private static final Logger log = LoggerFactory.getLogger(RegressionNetwork.class);
	private double learningRate = 0.07;
	private int epochs = 20;
	private Collection<? extends Collection<? extends Collection<Writable>>> initialData;
	private MultiLayerNetwork network;
	private int miniBatchSize = 50;
	private int miniBatchIterations = 1;
	private int hiddenLayerWidth = 10;
	private Collection<? extends Collection<? extends Collection<Writable>>> trainingData;
	private DataNormalization normalizer;
	private int numOfInputs = 1;

	protected RegressionNetwork() {
	}

	public static RegressionNetwork builder() {
		return new RegressionNetwork();
	}

	public RegressionNetwork build() {
		if (this.initialData == null) {
			throw new RuntimeException("no training data found");
		}
		this.trainingData = this.initialData;
		this.network = new MultiLayerNetwork(getNetworkConfiguration());
		this.network.init();
		this.network.setListeners(
				new IterationListener[] { new ScoreIterationListener(1000), new PerformanceListener(1000) });

		return this;
	}

	public MultiLayerNetwork train() {

		DataSetIterator trainIterator = getDataSetIterator(this.trainingData, this.miniBatchSize);
		normalizer = new NormalizerStandardize();
		normalizer.fit(trainIterator);
		trainIterator.reset();

		trainIterator.setPreProcessor(normalizer);

		for (int i = 0; i < this.epochs; i++) {
			this.network.fit(trainIterator);
			trainIterator.reset();
		}
		return this.network;
	}

	public float[] rnnTimeStep(int timeSteps, INDArray testingData) {
		return this.sampleFromNetwork(testingData, timeSteps);
	}

	public INDArray predict(int timeSteps) {
		INDArray input = Nd4j.create(new int[] { timeSteps, this.getNumOfInputs()});
		return this.network.output(input);
	}

	public DataSetIterator getDataSetIterator(Collection<? extends Collection<? extends Collection<Writable>>> trainingData, int miniBatchSize) {		
		SequenceRecordReader trainFeatures = new CollectionSequenceRecordReader(trainingData);
		SequenceRecordReaderDataSetIterator iterator = new SequenceRecordReaderDataSetIterator(trainFeatures,
				getMiniBatchSize(), 1, 1, true);
		return iterator;
	}

	public MultiLayerConfiguration getNetworkConfiguration() {
		MultiLayerConfiguration config = CommonNetworkConfigurations.recurrentNetwork(this.getNumOfInputs(), this.hiddenLayerWidth,
				getOutputCount(), getLearningRate(), getMiniBatchIterations());
		log.info(config.toYaml());
		return config;
	}
	
	/** Generate a sample from the network,
	 * can be used to 'prime' the RNN with a sequence you want to extend/continue.<br>
	 * Note that the initalization is used for all samples
	 * @param priori value,should be in the shape [1, numInputs, timeSteps]
	 */
	public float[] sampleFromNetwork(INDArray priori, int numTimeSteps){
		int inputCount = this.getNumOfInputs();
		float[] samples = new float[numTimeSteps];
		
		if(priori.size(1) != inputCount) {
			String format = String.format("the priori should have the same number of inputs [%s] as the trained network [%s]", priori.size(1), inputCount);
			throw new RuntimeException(format);
		}
		if(priori.size(2) < inputCount) {
			String format = String.format("the priori should have enough timesteps [%s] to prime the new inputs [%s]", priori.size(2), inputCount);
			throw new RuntimeException(format);
		}

		this.network.rnnClearPreviousState();
		INDArray output = this.network.rnnTimeStep(priori);
		
		output = output.ravel();
		// Store the output for use in the inputs
		LinkedList<Float> prevOutput = new LinkedList<>();
		for (int i = 0; i < output.length(); i++) {
			prevOutput.add(output.getFloat(0, i));
		}
		
		for( int i=0; i<numTimeSteps; ++i ){
			samples[i] = (prevOutput.peekLast());
			//Set up next input (single time step) by sampling from previous output
			INDArray nextInput = Nd4j.zeros(1,inputCount);
			
			float[] newInputs = new float[inputCount];
			newInputs[inputCount-1] = prevOutput.peekLast();
			for( int j=0; j<newInputs.length-1; j++ ) {
				newInputs[j] = prevOutput.get(prevOutput.size()-inputCount-j);
			}

			nextInput.assign(Nd4j.create(newInputs)); //Prepare next time step input
			output = this.network.rnnTimeStep(nextInput); //Do one time step of forward pass
			// Add the output to the end of the previous output queue
			prevOutput.addLast(output.ravel().getFloat(0, output.length()-1));
		}
		return samples;
	}

	private int getMiniBatchIterations() {
		return this.miniBatchIterations;
	}

	private double getLearningRate() {
		return this.learningRate;
	}

	public RegressionNetwork setLearningRate(double rate) {
		this.learningRate = rate;
		return this;
	}

	public Collection<? extends Collection<? extends Collection<Writable>>> getTrainingData() {
		return initialData;
	}

	public RegressionNetwork setTrainingData(Collection<? extends Collection<? extends Collection<Writable>>> trainingData2) {
		this.initialData = trainingData2;
		return this;
	}

	public int getEpochs() {
		return epochs;
	}

	public RegressionNetwork setEpochs(int iterations) {
		this.epochs = iterations;
		return this;
	}

	public int getOutputCount() {
		return 1;
	}

	public int getMiniBatchSize() {
		return miniBatchSize;
	}

	public int getNumOfInputs() {
		return numOfInputs;
	}

	public RegressionNetwork setNumOfInputs(int numOfInputs) {
		this.numOfInputs = numOfInputs;
		return this;
	}
}
