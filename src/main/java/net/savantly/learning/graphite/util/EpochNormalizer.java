package net.savantly.learning.graphite.util;

import org.joda.time.DateTime;

public class EpochNormalizer {

	/**
	 * 
	 * @param epoch millis
	 * @return a value that represents the day of week. Good for weekly cycles
	 */
    public static float standard(long epoch) {
    	DateTime date = new DateTime(epoch);
    	int dayOfWeek = date.getDayOfWeek();
    	int secondOfDay = date.getSecondOfDay();
		return new Float((dayOfWeek*100) + (secondOfDay * 0.001));
	}
    
    public static float standard(float epoch) {
    	return standard((long) epoch);
	}
    
}
