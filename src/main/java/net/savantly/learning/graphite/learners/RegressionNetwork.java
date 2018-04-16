package net.savantly.learning.graphite.learners;

import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedList;
import java.util.List;

import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionSequenceRecordReader;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.joda.time.DateTime;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import net.savantly.learning.graphite.util.EpochNormalizer;

public class RegressionNetwork {
	private static final Logger log = LoggerFactory.getLogger(RegressionNetwork.class);
	private static final int TIMESTEP_SEC = 15;
	private double learningRate = 0.001;
	private double rnnLearningRate = 0.001;
	private int epochs = 20;
	private Collection<? extends Collection<? extends Collection<Writable>>> initialData;
	private MultiLayerNetwork network;
	private int miniBatchSize = 1;
	private int miniBatchIterations = 1;
	private int hiddenLayerWidth = 30;
	private Collection<? extends Collection<? extends Collection<Writable>>> trainingData;
	private DataNormalization normalizer;
	private int numOfInputs = 1;
	private final List<IterationListener> iterationListeners = new ArrayList<>();

	protected RegressionNetwork() {
		this.iterationListeners.add(new PerformanceListener(1000));
		this.iterationListeners.add(new ScoreIterationListener(1000));
	}

	public static RegressionNetwork builder() {
		return new RegressionNetwork();
	}
	
	public RegressionNetwork build() {
		return build(new MultiLayerNetwork(getNetworkConfiguration()));
	}

	public RegressionNetwork build(MultiLayerNetwork network) {
		if (this.initialData == null) {
			throw new RuntimeException("no training data found");
		}
		this.trainingData = this.initialData;
		this.network = network;
		this.network.init();
		this.network.setListeners(this.getListeners());
		
		this.normalizer = new NormalizerStandardize();
		DataSetIterator trainIterator = getDataSetIterator(this.trainingData, this.miniBatchSize);
		normalizer.fitLabel(true);
		normalizer.fit(trainIterator);

		return this;
	}
	
	private IterationListener[] getListeners() {
		return this.iterationListeners.toArray(new IterationListener[0]);
	}

	public MultiLayerNetwork train() {
		DataSetIterator trainIterator = getDataSetIterator(this.trainingData, this.miniBatchSize);
		trainIterator.setPreProcessor(normalizer);

		for (int i = 0; i < this.epochs; i++) {
			this.network.fit(trainIterator);
			trainIterator.reset();
		}
		return this.network;
	}

	/**
	 * 
	 * @param timeSteps
	 * @param testingData
	 * @param containsEpoch The last feature is an epoch
	 * @return
	 */
	public float[] rnnTimeStep(int timeSteps, INDArray testingData, boolean containsEpoch) {
		return this.sampleFromNetwork(testingData, timeSteps, containsEpoch);
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
				getOutputCount(), getLearningRate(), getMiniBatchIterations(), this.getRnnLearningRate());
		log.info(config.toYaml());
		return config;
	}
	

	/** Generate a sample from the network,
	 * can be used to 'prime' the RNN with a sequence you want to extend/continue.<br>
	 * Note that the initalization is used for all samples
	 * @param priori value,should be in the shape [1, numInputs, timeSteps]
	 */
	public float[] sampleFromNetwork(INDArray priori, int numTimeSteps, boolean containsEpoch){
		// Log pre-normalized data
		if(log.isDebugEnabled()) {
			log.debug("pre-normalized test data:");
			for (int i = 0; i < priori.size(1); i++) {
				log.debug("[");
				for (int j = 0; j < priori.size(2); j++) {
					log.debug("{}", priori.getFloat(new int[] {0, i, j}));
				}
				log.debug("]");
			}
		}
		// Normalize test data
		this.normalizer.transform(priori);
		// log post-normalized data
		if(log.isDebugEnabled()) {
			log.debug("post-normalized test data:");
			for (int i = 0; i < priori.size(1); i++) {
				log.debug("[");
				for (int j = 0; j < priori.size(2); j++) {
					log.debug("{}", priori.getFloat(new int[] {0, i, j}));
				}
				log.debug("]");
			}
		}

		int inputCount = this.getNumOfInputs();
		INDArray samples = Nd4j.create(new int[] {numTimeSteps, 1});
		
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
		this.normalizer.revertFeatures(output);
		output = output.ravel();
		// Store the output for use in the inputs
		LinkedList<Float> prevOutput = new LinkedList<>();
		for (int i = 0; i < output.length(); i++) {
			prevOutput.add(output.getFloat(0, i));
		}
		
		for (int i=0; i<numTimeSteps; ++i){
			samples.putScalar(i,  0, prevOutput.peekLast());
			//Set up next input (single time step) by sampling from previous output
			INDArray nextInput = Nd4j.zeros(1,inputCount);

			float[] newInputs = new float[inputCount];
			
			// If there is an epoch feature, we need to keep it in the right place
			if (containsEpoch) {
				// calculate the current timestep seconds offset and normalize the value
				float epoch = EpochNormalizer.standard(DateTime.now().plusSeconds((i+1) * TIMESTEP_SEC).getMillis());
				newInputs[inputCount-1] = epoch;
				newInputs[inputCount-2] = prevOutput.peekLast();
				for( int j=0; j<newInputs.length-2; j++ ) {
					newInputs[j] = prevOutput.get(prevOutput.size()-1-j);
				}
			} else { // standard sliding [lag] window
				newInputs[inputCount-1] = prevOutput.peekLast();
				for( int j=0; j<newInputs.length-1; j++ ) {
					newInputs[j] = prevOutput.get(prevOutput.size()-1-j);
				}
			}

			nextInput.assign(Nd4j.create(newInputs)); //Prepare next time step input
			this.normalizer.transform(nextInput); // normalize the new features
			output = this.network.rnnTimeStep(nextInput); //Do one time step of forward pass
			this.normalizer.revertLabels(output); // revert the output
			// Add the output to the end of the previous output queue
			prevOutput.addLast(output.ravel().getFloat(0, output.length()-1));
		}
		// this.normalizer.revertLabels(samples);
		float[] result = new float[samples.rows()];
		for (int i = 0; i < result.length; i++) {
			result[i] = samples.getFloat(i, 0);
		}
		return result;
	}
	
	public void update(INDArray input) {
		boolean training = true;
		boolean storeLastForTBPTT = true;
		this.network.rnnActivateUsingStoredState(input, training, storeLastForTBPTT);
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

	public RegressionNetwork setHiddenLayerWidth(int hiddenLayerWidth) {
		this.hiddenLayerWidth = hiddenLayerWidth;
		return this;
	}

	public RegressionNetwork addIterationListener(IterationListener listener) {
		this.iterationListeners.add(listener);
		return this;
	}

	public RegressionNetwork setIterationListeners(Collection<IterationListener> listeners) {
		this.iterationListeners.clear();
		this.iterationListeners.addAll(listeners);
		return this;
	}

	public double getRnnLearningRate() {
		return this.rnnLearningRate;
	}
	
	public RegressionNetwork setRnnLearningRate(double rate) {
		this.rnnLearningRate = rate;
		return this;
	}
}
