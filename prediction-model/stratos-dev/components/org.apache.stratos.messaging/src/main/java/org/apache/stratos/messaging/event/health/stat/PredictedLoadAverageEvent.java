/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.stratos.messaging.event.health.stat;

import org.apache.stratos.messaging.event.Event;

public class PredictedLoadAverageEvent extends Event {
    private final String networkPartitionId;
    private final String clusterId;
    private final String clusterInstanceId;
    private final double value;
    private final String predictedValues;


    public PredictedLoadAverageEvent(String networkPartitionId, String clusterId,
                                     String clusterInstanceId, double value,String predictedValues) {
        this.networkPartitionId = networkPartitionId;
        this.clusterId = clusterId;
        this.clusterInstanceId = clusterInstanceId;
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

    public String getNetworkPartitionId() {
        return networkPartitionId;
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


}
