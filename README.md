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


## Example 

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
        
        
