package net.savantly.learning.graphite.learners.timeseries;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.joda.time.DateTime;
import org.nd4j.linalg.api.ndarray.INDArray;

import com.google.common.base.Strings;

import net.savantly.graphite.CarbonMetric;
import net.savantly.graphite.QueryableGraphiteClient;
import net.savantly.graphite.impl.SimpleCarbonMetric;
import net.savantly.graphite.query.GraphiteQuery;
import net.savantly.graphite.query.GraphiteQueryBuilder;
import net.savantly.graphite.query.fomat.CsvFormatter;
import net.savantly.learning.graphite.domain.GraphiteCsv;

public class PersistentGraphitePredictor {
	
	private QueryableGraphiteClient client;
	private String trainingTarget;
	private String testingTarget;
	private String targetAlias; 
	private GraphitePredictor graphitePredictor;
	private MultiLayerNetwork network;
	private int windowSize = 1;
	private List<IterationListener> iterationListeners = new ArrayList<>();
	private File dataDir;
	private boolean useCache = false;
	private double rnnLearningRate = 0.001;
	private double learningRate = 0.001;
	private int epochs = 50;
	private int hiddenLayerWidth = 30;
	private int tbpttLength = 1000;
	private int predictedStepCount = 2;

	private PersistentGraphitePredictor(QueryableGraphiteClient client) {
		this.client = client;
	}
	
	public static PersistentGraphitePredictor builder(QueryableGraphiteClient client) {
		return new PersistentGraphitePredictor(client);
	}
	
	public PersistentGraphitePredictor build() {
		if(this.dataDir == null) {
			try {
				this.dataDir = Files.createDirectories(Paths.get("data")).toFile();
			} catch (IOException e) {
				throw new RuntimeException(e);
			}
		}
		this.graphitePredictor = GraphitePredictor.builder()
				.setClient(client)
				.setEpochs(this.getEpochs())
				.setMiniBatchSize(1)
				.setLearningRate(this.getLearningRate())
				.setRnnLearningRate(this.getRnnLearningRate())
				.setNumOfInputs(windowSize+1)
				.setWindowSize(windowSize)
				.setTbpttLength(tbpttLength)
				.setHiddenLayerWidth(this.getHiddenLayerWidth());
		for (IterationListener iterationListener : iterationListeners) {
			this.graphitePredictor.addIterationListener(iterationListener);
		}
		return this;
	}

	private double getRnnLearningRate() {
		return this.rnnLearningRate;
	}

	private int getEpochs() {
		return this.epochs;
	}

	private double getLearningRate() {
		return this.learningRate;
	}

	private int getHiddenLayerWidth() {
		return this.hiddenLayerWidth;
	}

	public PersistentGraphitePredictor setTrainingTarget(String query) {
		this.trainingTarget = query;
		return this;
	}
	
	public PersistentGraphitePredictor setTestingTarget(String query) {
		this.testingTarget = query;
		return this;
	}
	
	public void train() {
		boolean saveUpdater = true;
		if(pretrainedNetworkExists() && this.isUseCache()) {
			try {
				MultiLayerNetwork existingNetwork = 
						ModelSerializer.restoreMultiLayerNetwork(this.getNetworkPath().toFile(), saveUpdater);
				this.graphitePredictor.setTrainingQuery(getTrainingQuery())
				.build(existingNetwork);
			} catch (IOException e) {
				throw new RuntimeException(e);
			}
			
		} else {
			this.graphitePredictor.setTrainingQuery(getTrainingQuery())
			.build();
			this.network = this.graphitePredictor.train();
			try {
				ModelSerializer.writeModel(this.network, getNetworkPath().toFile(), saveUpdater);
			} catch (IOException e) {
				throw new RuntimeException(e);
			}
		}
	}
	
	private boolean pretrainedNetworkExists() {
		return Files.exists(getNetworkPath());
	}

	private Path getNetworkPath() {
		return Paths.get(this.dataDir.getPath(), this.targetToSafeName() + ".network");
	}
	
	public void update() {
		//this.updateNetwork();
		float[] predictions = this.predict(predictedStepCount ); // 15sec increments
		List<CarbonMetric> metrics = new ArrayList<>();
		for (int i = 0; i<predictedStepCount; i++) {
			String metricName = this.getPredictedTarget();
			String value = String.format("%s", predictions[i]);
			// we offset the epoch by the number of steps we predicted
			// since we cannot store future dated metrics
			long epoch = DateTime.now().minusSeconds(predictedStepCount*15).plusSeconds(15*(i)).getMillis();
			metrics.add(new SimpleCarbonMetric(metricName, value, epoch/1000));
		}
		this.client.saveCarbonMetrics(metrics);
	}

	public void updateNetwork() {
		String queryResult = this.client.query(this.getUpdateQuery());
		INDArray priori = GraphiteCsv.from(queryResult).as3dMatrix(this.windowSize, false);
		this.graphitePredictor.update(priori);
	}
	
	private GraphiteQuery<String> getTrainingQuery() {
		GraphiteQueryBuilder<String> builder = new GraphiteQueryBuilder<>(new CsvFormatter());
		return builder.setFrom("-168h")
				.setUntil("-15min")
				.setTarget(trainingTarget)
				.build();
	}
	
	private GraphiteQuery<String> getTestingQuery() {
		GraphiteQueryBuilder<String> builder = new GraphiteQueryBuilder<>(new CsvFormatter());
		return builder.setFrom("-45min")
				.setUntil("-1min")
				.setTarget(testingTarget)
				.build();
	}
	
	private GraphiteQuery<String> getUpdateQuery() {
		GraphiteQueryBuilder<String> builder = new GraphiteQueryBuilder<>(new CsvFormatter());
		return builder.setFrom("-15min")
				.setUntil("-1min")
				.setTarget(trainingTarget)
				.build();
	}
	
	public Function<String[], Boolean> filterZeros() {
		return (parts) -> {
			if (Strings.isNullOrEmpty(parts[2])) {
				return false;
			} else { 
				float parsed = Float.parseFloat(parts[2]);
				return parsed > 0;
			}
		};
	}

	public float[] predict(int stepCount) {
		String queryResult = this.client.query(this.getTestingQuery());
		INDArray priori = GraphiteCsv.from(queryResult).as3dMatrix(this.windowSize, false);
		return this.graphitePredictor.doTimeStep(stepCount, priori);
	}
	
	private String targetToSafeName() {
		return this.trainingTarget.replaceAll("[^a-zA-Z0-9]", "");
	}
	
	private String getPredictedTarget() {
		if(this.targetAlias == null) {
			throw new RuntimeException("the targetAlias must be set, otherwise the predictions will not be stored correctly");
		}
		return String.format("%s_predicted", this.targetAlias);
	}

	public PersistentGraphitePredictor setWindowSize(int i) {
		this.windowSize = i;
		return this;
	}
	
	public PersistentGraphitePredictor addIterationListener(IterationListener listener) {
		this.iterationListeners.add(listener);
		return this;
	}

	public File getDataDir() {
		return dataDir;
	}

	public PersistentGraphitePredictor setDataDir(File dataDir) {
		this.dataDir = dataDir;
		return this;
	}

	public boolean isUseCache() {
		return useCache;
	}

	public PersistentGraphitePredictor setUseCache(boolean useCache) {
		this.useCache = useCache;
		return this;
	}

	public String getTargetAlias() {
		return targetAlias;
	}

	public PersistentGraphitePredictor setTargetAlias(String targetAlias) {
		this.targetAlias = targetAlias;
		return this;
	}

	public PersistentGraphitePredictor setRnnLearningRate(double rnnLearningRate) {
		this.rnnLearningRate = rnnLearningRate;
		return this;
	}

	public PersistentGraphitePredictor setLearningRate(double learningRate) {
		this.learningRate = learningRate;
		return this;
	}

	public PersistentGraphitePredictor setEpochs(int epochs) {
		this.epochs = epochs;
		return this;
	}

	public PersistentGraphitePredictor setHiddenLayerWidth(int hiddenLayerWidth) {
		this.hiddenLayerWidth = hiddenLayerWidth;
		return this;
	}

	public int getTbpttLength() {
		return tbpttLength;
	}

	public PersistentGraphitePredictor setTbpttLength(int tbpttLength) {
		this.tbpttLength = tbpttLength;
		return this;
	}

	public int getPredictedStepCount() {
		return predictedStepCount;
	}

	public PersistentGraphitePredictor setPredictedStepCount(int predictedStepCount) {
		this.predictedStepCount = predictedStepCount;
		return this;
	}

}
