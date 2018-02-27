# graphite-dl4j

Perform machine algorithms on graphite query results.  

You can use the QueryableGraphiteClient to get positive and negative condition examples to load the classifier.  

[graphite client](https://github.com/savantly-net/graphite-client)  


## Usage 


	<dependency>
		<groupId>net.savantly.learning</groupId>
		<artifactId>graphite-dl4j</artifactId>
		<version>1.0.0-RELEASE</version>
	</dependency>


## Quickstart 
add graphite-dl4j as a dependency to your project.  

This example uses a preconfigured multilayer network to classify a graphite query result as either positive or negative, 
based on the examples provided to the builder -  

		GraphiteBooleanClassification classifier = GraphiteBooleanClassification.builder()
				.setWorkingDirectory(Files.createDirectories(Paths.get("target/data")).toFile())
				.addNegativeQuery(GraphiteQueryBuilder.simpleQuery("alias(constantLine(-1), 'A')"))
				.addNegativeQuery(GraphiteQueryBuilder.simpleQuery("alias(constantLine(-1), 'B')"))
				.addPositiveQuery(GraphiteQueryBuilder.simpleQuery("alias(constantLine(1), 'A')"))
				.addPositiveQuery(GraphiteQueryBuilder.simpleQuery("alias(constantLine(1), 'B')"))
				.setNumberOfIterations(5)
				.setClient(GraphiteClientFactory.queryableGraphiteClient(graphiteHost))
				.build();
		MultiLayerNetwork network = classifier.train();
		log.info(network.summary());
		
The trained network can then be used to predict the class of a new graphite query result -  

    network.test


## Examples 

### Get json data from graphite query  

#### using the graphite client

	JsonFormatter formatter = new JsonFormatter();
	GraphiteQueryBuilder<JsonNode> builder = new GraphiteQueryBuilder<>(formatter);
	GraphiteQuery<JsonNode> query = builder.setTarget(target).build();
	
	JsonNode json = client.query(query); 
	
### transform json to MultiSeriesGraphite  

		List<GraphiteMultiSeries> positiveTrainingExamples = new ArrayList<>();
		List<GraphiteMultiSeries> negativeTrainingExamples = new ArrayList<>();

		// Add the 'negative condition' examples
		Arrays.stream(this.goodies).forEach(g -> {
			try {
				negativeTrainingExamples.add(GraphiteMultiSeries.from(g)); // load from InputString, json, etc...
			} catch (IOException e) {
				log.error("{}", e);
			}
		});

		// Add the 'positive condition' examples
		Arrays.stream(this.baddies).forEach(g -> {
			try {
				positiveTrainingExamples.add(GraphiteMultiSeries.from(g)); // load from InputString, json, etc...
			} catch (IOException e) {
				log.error("{}", e);
			}
		});
		
#### Supply examples to classifier  

		TimeSeriesStateClassifier classifier = TimeSeriesStateClassifier.builder()
				.setWorkingDirectory(dir)
				.setPositiveExamples(positiveTrainingExamples)
				.setNegativeExamples(negativeTrainingExamples)
				.setNumberOfIterations(5)
				.build();
		
		MultiLayerNetwork result = classifier.train();
		
		//Save the model
        File locationToSave = new File("target/MyMultiLayerNetwork.zip");      //Where to save the network. Note: the file is in .zip format - can be opened externally
        boolean saveUpdater = true;                                             //Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
        ModelSerializer.writeModel(result, locationToSave, saveUpdater);
        
        
