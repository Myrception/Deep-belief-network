using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DeepBeliefNeuralNetwork.MLPComponents.Funktionen;

namespace DeepBeliefNeuralNetwork.MLPComponents
{
    internal static class MLPVectorCalculation
    {
        /// <summary>
        /// Methode zur Berechnung der Ausgabe des Neuronalen Netzwerkes ohne Bias
        /// </summary>
        /// <param name="inputvector">Der Eingabevektor</param>
        /// <param name="layers">Übergabe des Netzes</param>
        /// <param name="matrix">Die Gewichtsmatrix</param>
        /// <param name="alpha">Ein Term in den Funktionen womit diese beinflusst werden können</param>
        /// <returns></returns>
        internal static double[] CalculateTarget(this double[] inputvector, List<List<MLPNeuron>> layers, MLPWeightMatrix matrix ,double alpha)
        {
            bool softmaxusingoutput = false, softmaxusingactivation = false;
            double SumForSoftmaxClassifier = 0;
            double[] outputVector = new double[layers[layers.Count - 1].Count];
            int j = 0;
            foreach (List<MLPNeuron> singleLayer in layers)
            {
                foreach (MLPNeuron neuronOfLayer in singleLayer)
                {
                    neuronOfLayer.NetInput = 0;
                }
            }
            for (int i = 0; i < layers.Count; i++)
            {
                if (i == 0)//for the Input Layer
                {
                    for (int k = 0; k < inputvector.Length; k++)
                    {
                        layers[0][k].NetInput = inputvector[k];

                        layers[0][k].Calculate(alpha);

                    }
                }
                else // the other Layers
                {
                    foreach (MLPNeuron layerN in layers[i])
                    {
                        foreach (MLPNeuron layerNMinusOne in layers[i - 1])
                        {
                            if (double.IsNaN(matrix[layerNMinusOne.Index, layerN.Index]))
                            {
                                layerN.NetInput += 0;
                            }
                            else
                            {
                                layerN.NetInput += layerNMinusOne.Output * matrix[layerNMinusOne.Index, layerN.Index];
                                //layerN.NetInput=layerN.NetInput / 100d;
                            }
                            if (double.IsNaN(layerN.NetInput))
                            {
                                
                            }
                        }
                        layerN.Calculate(alpha);
                        if (layerN.OutputFunction is Softmax)
                        {
                            softmaxusingoutput = true;
                        }
                        if (layerN.ActivationFunction is Softmax)
                        {
                            softmaxusingactivation = true;
                        }
                        if (i == layers.Count - 1)
                        {
                            SumForSoftmaxClassifier += Math.Exp(layerN.NetInput);
                        }

                    }
                    if (softmaxusingoutput) 
                    {
                    foreach (MLPNeuron layerN in layers[i])
                        {
                        //layerN.calculate(alpha);
                            layerN.Output = layerN.Output / SumForSoftmaxClassifier;
                            if (double.IsNaN(layerN.Output))
                            {
                                
                            }
                        }
                    }
                    else if (softmaxusingactivation)
                    {
                        foreach (MLPNeuron layerN in layers[i])
                        {
                            //layerN.calculate(alpha);
                            layerN.Output = layerN.Output / SumForSoftmaxClassifier;
                            if (double.IsNaN(layerN.Output))
                            {

                            }
                        }
                    }
                }
            }
            foreach (MLPNeuron outputLayer in layers[layers.Count - 1])
            {
                outputVector[j] = outputLayer.Output;
                j++;
            }
            return outputVector;
        }
        /// <summary>
        /// Methode zur Berechnung der Ausgabe des Neuronalen Netzwerkes mit Bias
        /// </summary>
        /// <param name="inputvector">Der Eingabevektor</param>
        /// <param name="layers">Übergabe des Netzes</param>
        /// <param name="matrix">Die Gewichtsmatrix</param>
        /// <param name="RBMBias">Liste von den Bias Neuronen</param>
        /// <param name="alpha">Ein Term in den Funktionen womit diese beinflusst werden können</param>
        /// <returns></returns>
        internal static double[] CalculateTargetWithBias(this double[] inputvector, List<List<MLPNeuron>> layers, MLPWeightMatrix matrix, List<RBMComponents.RBMBiasNeuron> RBMBias, double alpha)
        {
            bool softmaxusingoutput = false, softmaxusingactivation = false;
            double SumForSoftmaxClassifier = 0;
            double[] outputVector = new double[layers[layers.Count - 1].Count];
            int j = 0;
            foreach (List<MLPNeuron> singleLayer in layers)
            {
                foreach (MLPNeuron neuronOfLayer in singleLayer)
                {
                    neuronOfLayer.NetInput = 0;
                }
            }
            for (int i = 0; i < layers.Count; i++)
            {
                if (i == 0)//for the Input Layer
                {
                    for (int k = 0; k < inputvector.Length; k++)
                    {
                        layers[0][k].NetInput = inputvector[k];

                        layers[0][k].Calculate(alpha);
                        if (double.IsNaN(layers[0][k].NetInput))
                        {

                        }

                    }
                }
                else // the other Layers
                {
                    foreach (MLPNeuron layerN in layers[i])
                    {
                        foreach (MLPNeuron layerNMinusOne in layers[i - 1])
                        {
                            if (i==layers.Count - 1)
                            {
                                layerN.NetInput += (layerNMinusOne.Output * matrix[layerNMinusOne.Index, layerN.Index]) + 0d;
                            }
                            else
                            {
                                layerN.NetInput += (layerNMinusOne.Output * matrix[layerNMinusOne.Index, layerN.Index]) + RBMBias[i-1].BiasHidden;
                               
                            }
                            if (double.IsNaN(layerN.NetInput))
                            {

                            }
                        }

                        layerN.Calculate(alpha);
                        if (layerN.OutputFunction is Softmax)
                        {
                            softmaxusingoutput = true;
                        }
                        if (layerN.ActivationFunction is Softmax)
                        {
                            softmaxusingactivation = true;
                        }
                        if (i == layers.Count - 1)
                        {
                            SumForSoftmaxClassifier += Math.Exp(layerN.NetInput);
                        }

                    }
                    if (softmaxusingoutput)
                    {
                        foreach (MLPNeuron layerN in layers[i])
                        {
                            //layerN.calculate(alpha);
                            layerN.Output = layerN.Output / SumForSoftmaxClassifier;
                            if (double.IsNaN(layerN.Output))
                            {

                            }
                        }
                    }
                    else if (softmaxusingactivation)
                    {
                        foreach (MLPNeuron layerN in layers[i])
                        {
                            //layerN.calculate(alpha);
                            layerN.Output = layerN.Output / SumForSoftmaxClassifier;
                            if (double.IsNaN(layerN.Output))
                            {

                            }
                        }
                    }
                }
            }
            foreach (MLPNeuron outputLayer in layers[layers.Count - 1])
            {
                outputVector[j] = outputLayer.Output;
                j++;
                if (double.IsNaN(outputLayer.Output))
                {

                }
            }
            return outputVector;
        }
    }
}
