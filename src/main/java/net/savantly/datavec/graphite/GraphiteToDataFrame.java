package net.savantly.datavec.graphite;

import java.util.Arrays;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.transform.schema.Schema;

public class GraphiteToDataFrame {
	
	public GraphiteToDataFrame() {
        //Let's define the schema of the data that we want to import
        //The order in which columns are defined here should match the order in which they appear in the input data
        Schema inputDataSchema = new Schema.Builder()
            //We can define a single column
            .addColumnString("DateTimeString")
            //Or for convenience define multiple columns of the same type
            .addColumnsString("CustomerID", "MerchantID")
            //We can define different column types for different types of data:
            .addColumnInteger("NumItemsInTransaction")
            .addColumnCategorical("MerchantCountryCode", Arrays.asList("USA","CAN","FR","MX"))
            //Some columns have restrictions on the allowable values, that we consider valid:
            .addColumnDouble("TransactionAmountUSD",0.0,null,false,false)   //$0.0 or more, no maximum limit, no NaN and no Infinite values
            .addColumnCategorical("FraudLabel", Arrays.asList("Fraud","Legit"))
            .build();

        //Print out the schema:
        System.out.println("Input data schema details:");
        System.out.println(inputDataSchema);

        System.out.println("\n\nOther information obtainable from schema:");
        System.out.println("Number of columns: " + inputDataSchema.numColumns());
        System.out.println("Column names: " + inputDataSchema.getColumnNames());
        System.out.println("Column types: " + inputDataSchema.getColumnTypes());
	}
	
	public void readRecords() {
		RecordReader reader;
	}

}
