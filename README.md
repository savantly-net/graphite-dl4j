# graphite-dl4j

Perform machine algorithms on graphite query results.  

You can use the QueryableGraphiteClient to get positive and negative condition examples to load the classifier.  

[graphite client](https://github.com/savantly-net/graphite-client)  


## Usage 


	<dependency>
		<groupId>net.savantly.learning</groupId>
		<artifactId>graphite-dl4j</artifactId>
		<version>2.0.0-SNAPSHOT</version>
	</dependency>


## Quickstart 
add graphite-dl4j as a dependency to your project.  

This example uses a preconfigured multilayer network to train and predict -  
	
	private static final String graphiteHost = "127.0.0.1";
	private static final int windowSize = 2;
	private static final String target = "randomWalk()";
	private static final String targetAlias = "random.walk";
	private static final int epochs = 20;
	private static final int hiddenLayerWidth = 30;
	private static final double learningRate = 0.001;

	public void testTraining() throws UnknownHostException, SocketException {
		QueryableGraphiteClient client = GraphiteClientFactory.queryableGraphiteClient(graphiteHost);
		PersistentGraphitePredictor learner = 
				PersistentGraphitePredictor.builder(client )
				.setWindowSize(windowSize)
				.setTrainingTarget(target)
				.setTestingTarget(target)
				.setTargetAlias(targetAlias)
				.setEpochs(epochs)
				.setHiddenLayerWidth(hiddenLayerWidth)
				.setLearningRate(learningRate)
				.setRnnLearningRate(learningRate)
				.setUseCache(false) // you can cache/store the model to re-use between app restarts
				.build();
		
		learner.train(); // train the network using the graphite targets
		learner.update(); // make a 2 min prediction and store it in graphite
	}


## Lower level configuration  

		this.graphitePredictor = GraphitePredictor.builder()
				.setClient(client)
				.setEpochs(this.getEpochs())
				.setMiniBatchSize(1)
				.setLearningRate(this.getLearningRate())
				.setRnnLearningRate(this.getRnnLearningRate())
				.setNumOfInputs(windowSize+1)
				.setWindowSize(windowSize)
				.setHiddenLayerWidth(this.getHiddenLayerWidth());
				
## Even lower...

		this.network = RegressionNetwork.builder()
				.setEpochs(epochs)
				.setLearningRate(learningRate)
				.setRnnLearningRate(rnnLearningRate)
				.setTrainingData(trainingData)
				.setNumOfInputs(numOfInputs)
				.setHiddenLayerWidth(hiddenLayerWidth)
				.setIterationListeners(iterationListeners);




