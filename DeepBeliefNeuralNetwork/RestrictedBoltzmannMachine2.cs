using DeepBeliefNeuralNetwork.MLPComponents;
using DeepBeliefNeuralNetwork.RBMComponents;
using System;
using System.Collections.Generic;

namespace DeepBeliefNeuralNetwork
{
    internal class RestrictedBoltzmannMachine2
    {
        public List<List<RBMNeurons>> Layers = new List<List<RBMNeurons>>();
        public List<RBMBiasNeuron> Bias = new List<RBMBiasNeuron>();
        public RBMWeightMatrix Matrix { get; set; }

        private int _weightTemp1 = 0, _weightTemp2 = 0; //Hilfsvariablen zur Bestimmung der Richtigen Positionen der Gewichtsänderungen

        public RestrictedBoltzmannMachine2(int[] neurons, Random RND)
        {
            int index = 0, count = 0;
            //int numToRemove = 10;
            int[] neuron = new int[neurons.Length - 1];
            //neuron = neurons.Where(val => val != numToRemove).ToArray();
            for (int i = 0; i < neurons.Length - 1; i++)
            {
                neuron[i] = neurons[i];
            }
            foreach (var item in neuron)
            {
                count += item;
                List<RBMNeurons> temp = new List<RBMNeurons>();
                for (int i = 0; i < item; i++)
                {
                    temp.Add(new RBMNeurons() { Index = index });
                    index++;
                }
                Layers.Add(temp);
            }
            for (int i = 0; i < neurons.Length - 1; i++)
            {
                Bias.Add(new RBMBiasNeuron());
            }
            Matrix = new RBMWeightMatrix(count, count);
            //CloneMatrix = new double[count, count];
            WeightMatrixRandom(neuron, RND);
        }

        public void GreedyTrainingRestrictedBoltzmannMachine(double[] inputvector, int contrastiveDivergence, double learningRate, Random rnd)
        {
            int temp = 0, temp2 = 0, temp4 = Layers[0].Count; // temp3=0,
            _weightTemp1 = 0;
            _weightTemp2 = 0;
            double[] _initialBinaryHiddenList = new double[0];
            for (int i = 0; i < inputvector.Length; i++)
            {
                Layers[0][i].BinaryState = inputvector[i];
            }
            for (int i = 0; i < Layers.Count - 1; i++)
            {
                temp2 += Layers[i].Count;
                //temp3 += Layers[i + 1].Count;
                temp4 += Layers[i + 1].Count;
                foreach (var hiddenNeuron in Layers[i + 1])//calculate hidden
                {
                    double sum = 0;

                    for (int j = temp; j < temp2; j++)
                    {
                        sum += Layers[i][j - temp].BinaryState * Matrix[j, hiddenNeuron.Index];
                    }

                    double probability = 1d / (1d + Math.Exp(-(Bias[i].BiasHidden + sum)));
                    hiddenNeuron.Probability = probability;
                    hiddenNeuron.BinaryState = rnd.NextDouble() <= probability ? 1 : 0;
                }
                _initialBinaryHiddenList = new double[Layers[i + 1].Count];
                for (int j = 0; j < Layers[i + 1].Count; j++)
                {
                    _initialBinaryHiddenList[j] = Layers[i + 1][j].Probability;
                }
                for (int cd = 0; cd < contrastiveDivergence; cd++)
                {
                    //Console.WriteLine("CD:" + cd);
                    foreach (var visibleNeuron in Layers[i])//calculate reconstruction driven by Data
                    {
                        double sum = 0;
                        for (int j = temp2; j < temp4; j++)
                        {
                            sum += Layers[i + 1][j - temp2].BinaryState * Matrix[j, visibleNeuron.Index];
                        }
                        double probability = 1d / (1d + Math.Exp(-(Bias[i].BiasVisible + sum)));
                        visibleNeuron.Probability = probability;
                        visibleNeuron.BinaryState = rnd.NextDouble() <= probability ? 1 : 0;
                    }
                    if (i == cd - 1)
                    {
                        foreach (var hiddenNeuron in Layers[i + 1])//calculate hidden with reconstructed Data for the last time
                        {
                            double sum = 0;

                            for (int j = temp; j < temp2; j++)
                            {
                                sum += Layers[i][j - temp].Probability * Matrix[j, hiddenNeuron.Index];
                            }
                            double probability = 1d / (1d + Math.Exp(-(Bias[i].BiasHidden + sum)));
                            hiddenNeuron.Probability = probability;
                            hiddenNeuron.BinaryState = rnd.NextDouble() <= probability ? 1 : 0;
                        }
                    }
                    else
                    {
                        foreach (var hiddenNeuron in Layers[i + 1])//calculate hidden with reconstructed Data
                        {
                            double sum = 0;

                            for (int j = temp; j < temp2; j++)
                            {
                                sum += Layers[i][j - temp].BinaryState * Matrix[j, hiddenNeuron.Index];
                            }
                            double probability = 1d / (1d + Math.Exp(-(Bias[i].BiasHidden + sum)));
                            hiddenNeuron.Probability = probability;
                            hiddenNeuron.BinaryState = rnd.NextDouble() <= probability ? 1 : 0;
                        }
                    }
                }
                CalculateWeightMatrixAlteration(learningRate, inputvector, _initialBinaryHiddenList, Layers[i],
                    Layers[i + 1], Bias[i]);
                foreach (var hiddenNeuron in Layers[i + 1])//calculate hidden after weight alterration for the next layer
                {
                    double sum = 0;

                    for (int j = temp; j < temp2; j++)
                    {
                        sum += Layers[i][j - temp].BinaryState * Matrix[j, hiddenNeuron.Index];
                    }

                    double probability = 1d / (1d + Math.Exp(-(Bias[i].BiasHidden + sum)));
                    hiddenNeuron.Probability = probability;
                    hiddenNeuron.BinaryState = rnd.NextDouble() <= probability ? 1 : 0;
                }
                inputvector = new double[Layers[i + 1].Count];
                for (int j = 0; j < Layers[i + 1].Count; j++)
                {
                    inputvector[j] = Layers[i + 1][j].BinaryState;
                }
                temp += Layers[i].Count;
            }
        }

        public void CalculateWeightMatrixAlteration(double learningRate, double[] inputvector, double[] initialBinaryHiddenList, List<RBMNeurons> actualVisible, List<RBMNeurons> actualHidden, RBMBiasNeuron actualBias)
        {
            _weightTemp2 += actualVisible.Count;
            for (int i = 0; i < actualVisible.Count; i++) // Update weightmatrix
            {
                for (int j = _weightTemp2; j < _weightTemp2 + actualHidden.Count; j++)
                {
                    Matrix[i + _weightTemp1, j] += (learningRate *
                                                       ((inputvector[i] *
                                                         initialBinaryHiddenList[j - _weightTemp2]) -
                                                        (actualVisible[i].BinaryState *
                                                         actualHidden[j - _weightTemp2].Probability)));
                    Matrix[j, i + _weightTemp1] = Matrix[i + _weightTemp1, j];
                }
            }
            _weightTemp1 += actualVisible.Count;
            //for (int i = 0; i < inputvector.Length; i++) // Update Bias
            //{
            //    actualBias.BiasVisible += (learningRate *
            //                              (inputvector[i] - actualVisible[i].Probability)) /
            //                             inputvector.Length;

            //}
            //for (int i = 0; i < actualHidden.Count; i++)
            //{
            //    actualBias.BiasHidden += (learningRate *
            //                             (initialBinaryHiddenList[i] - actualHidden[i].Probability)) /
            //                            actualHidden.Count;
            //}
        }

        public void GreedyTrainingRestrictedBoltzmannMachineWithPicture(double[] inputvector, double learningRate, Random rnd)
        {
            Bilderstellen bild;
            int temp = 0, temp2 = 0, temp3 = 0, temp4 = Layers[0].Count;
            _weightTemp1 = 0;
            _weightTemp2 = 0;
            double[] _initialBinaryHiddenList = new double[0];
            for (int i = 0; i < inputvector.Length; i++)
            {
                Layers[0][i].BinaryState = inputvector[i];
            }
            double picturesize = Math.Sqrt(inputvector.Length);
            bild = new Bilderstellen((int)picturesize, (int)picturesize);
            int counter = 0;
            for (int k = 0; k < 28; k++)
            {
                for (int j = 0; j < 28; j++)
                {
                    if (Layers[0][counter].BinaryState == 1) // ist keine Spirale Farbe Weiß
                    {
                        bild.BildPixelPlazieren(k, j, System.Drawing.Color.White);
                    }
                    if (Layers[0][counter].BinaryState == 0)
                    {
                        bild.BildPixelPlazieren(k, j, System.Drawing.Color.Black);
                    }
                    counter++;
                }
            }
            bild.SavePicture(@"C:\Users\Joseph\Desktop\" + "ursprungsbild" + "Bild" + ".png");
            for (int i = 0; i < Layers.Count - 1; i++)
            {
                temp2 += Layers[i].Count;
                temp3 += Layers[i + 1].Count;
                temp4 += Layers[i + 1].Count;
                foreach (var hiddenNeuron in Layers[i + 1])//calculate hidden
                {
                    double sum = 0;

                    for (int j = temp; j < temp2; j++)
                    {
                        sum += Layers[i][j - temp].BinaryState * Matrix[j, hiddenNeuron.Index];
                    }

                    double probability = 1d / (1d + Math.Exp(-(Bias[i].BiasHidden + sum)));
                    hiddenNeuron.Probability = probability;
                    hiddenNeuron.BinaryState = rnd.NextDouble() <= probability ? 1 : 0;
                }
                for (int cd = 0; cd < 1; cd++)
                {
                    if (cd == 0) //notwendig um nachher die Gewichtsänderungen zu berechnen
                    {
                        _initialBinaryHiddenList = new double[Layers[i + 1].Count];
                        for (int j = 0; j < Layers[i + 1].Count; j++)
                        {
                            _initialBinaryHiddenList[j] = Layers[i + 1][j].Probability;
                        }
                    }
                    foreach (var visibleNeuron in Layers[i])//calculate reconstruction driven by Data
                    {
                        double sum = 0;
                        for (int j = temp2; j < temp4; j++)
                        {
                            sum += Layers[i + 1][j - temp2].BinaryState * Matrix[j, visibleNeuron.Index];
                        }
                        double probability = 1d / (1d + Math.Exp(-(Bias[i].BiasVisible + sum)));
                        visibleNeuron.Probability = probability;
                        visibleNeuron.BinaryState = rnd.NextDouble() <= probability ? 1 : 0;
                    }
                    picturesize = Math.Sqrt(Layers[i].Count);
                    bild = new Bilderstellen((int)picturesize, (int)picturesize);
                    counter = 0;
                    for (int k = 0; k < (int)picturesize; k++)
                    {
                        for (int j = 0; j < (int)picturesize; j++)
                        {
                            if (Layers[0][counter].BinaryState == 1) // ist keine Spirale Farbe Weiß
                            {
                                bild.BildPixelPlazieren(k, j, System.Drawing.Color.White);
                            }
                            if (Layers[0][counter].BinaryState == 0)
                            {
                                bild.BildPixelPlazieren(k, j, System.Drawing.Color.Black);
                            }
                            counter++;
                        }
                    }
                    bild.SavePicture(@"C:\Users\Joseph\Desktop\" + i + "Bild" + ".png");
                    if (i == cd - 1)
                    {
                        foreach (var hiddenNeuron in Layers[i + 1])//calculate hidden with reconstructed Data for the last time
                        {
                            double sum = 0;

                            for (int j = temp; j < temp2; j++)
                            {
                                sum += Layers[i][j - temp].Probability * Matrix[j, hiddenNeuron.Index];
                            }
                            double probability = 1d / (1d + Math.Exp(-(Bias[i].BiasHidden + sum)));
                            hiddenNeuron.Probability = probability;
                            hiddenNeuron.BinaryState = rnd.NextDouble() <= probability ? 1 : 0;
                        }
                    }
                    else
                    {
                        foreach (var hiddenNeuron in Layers[i + 1])//calculate hidden with reconstructed Data
                        {
                            double sum = 0;

                            for (int j = temp; j < temp2; j++)
                            {
                                sum += Layers[i][j - temp].BinaryState * Matrix[j, hiddenNeuron.Index];
                            }
                            double probability = 1d / (1d + Math.Exp(-(Bias[i].BiasHidden + sum)));
                            hiddenNeuron.Probability = probability;
                            hiddenNeuron.BinaryState = rnd.NextDouble() <= probability ? 1 : 0;
                        }
                    }
                }
                foreach (var hiddenNeuron in Layers[i + 1])//calculate hidden after weight alterration for the next layer
                {
                    double sum = 0;

                    for (int j = temp; j < temp2; j++)
                    {
                        sum += Layers[i][j - temp].BinaryState * Matrix[j, hiddenNeuron.Index];
                    }

                    double probability = 1d / (1d + Math.Exp(-(Bias[i].BiasHidden + sum)));
                    hiddenNeuron.Probability = probability;
                    hiddenNeuron.BinaryState = rnd.NextDouble() <= probability ? 1 : 0;
                }
                inputvector = new double[Layers[i + 1].Count];
                for (int j = 0; j < Layers[i + 1].Count; j++)
                {
                    inputvector[j] = Layers[i + 1][j].BinaryState;
                }
                temp += Layers[i].Count;
            }
        }

        private void WeightMatrixRandom(int[] Neurons, Random RND)
        {
            int temp = 0, temp2 = 0;
            IFunktionen Funktion = new MLPComponents.Funktionen.SigmoideFunktion();
            for (int l = 0; l < Neurons.Length - 1; l++)
            {
                XavierInitialization Xavier = new XavierInitialization(Neurons[l],
                    Neurons[l + 1], Funktion);
                temp2 += Neurons[l];
                for (int j = 0; j < Neurons[l]; j++)
                {
                    for (int i = temp2; i < temp2 + Neurons[l + 1]; i++)
                    {
                        //Matrix[j + temp, i] = Xavier.CalculateRandom(RND);
                        Matrix[j + temp, i] = 2 * (RND.NextDouble() - 0.5);
                        Matrix[i, j + temp] = Matrix[j + temp, i];
                    }
                }
                temp += Neurons[l];
            }
        }
    }
}