package org.apache.stratos.messaging.event.health.stat;

import org.apache.stratos.messaging.event.Event;

public class MemberPredictedMemoryConsumptionEvent extends Event{
	private final String clusterId;
	private final String clusterInstanceId;
	private final String memberId;
	private final double value;
	private final String predictedValues;
	public MemberPredictedMemoryConsumptionEvent(String clusterId, String clusterInstanceId,
	                                       String memberId, double value,String predictedValues) {
		this.clusterId = clusterId;
		this.clusterInstanceId = clusterInstanceId;
		this.memberId = memberId;
		this.value = value;
		this.predictedValues=predictedValues;

	}


	private   double[] decodePredictions(String s)
	{

		if(s!=null && s.contains(",")) {
			String[] temp = s.split(",");
			double []predictions = new double[temp.length];
			for (int i = 0; i < temp.length; i++) {
				predictions[i] = Double.parseDouble(temp[i]);
			}
			return predictions;
		}

		return null;
	}

	public String getClusterId() {
		return clusterId;
	}

	public String getClusterInstanceId() {
		return clusterInstanceId;
	}

	public double getValue() {
		return value;
	}

	public double[] getPredictions(){
		return decodePredictions(predictedValues);
	}


	public String getMemberId() {
		return memberId;
	}

	public String getPredictedValues() {
		return predictedValues;
	}
}
