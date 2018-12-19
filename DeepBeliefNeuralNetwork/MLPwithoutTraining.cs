using DeepBeliefNeuralNetwork.MLPComponents;
using DeepBeliefNeuralNetwork.MLPComponents.Funktionen;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepBeliefNeuralNetwork
{
    public class MLPwithoutTraining
    {
        List<RestrictedBoltzmannMachine> RBMLayer = new List<RestrictedBoltzmannMachine>();
        List<MLPCreateNeuralNetwork> Test2 = new List<MLPCreateNeuralNetwork>();
        RestrictedBoltzmannMachine2 RBM2;
        MultiLayerPerceptron2 TestNetz2;
        Random RND = new Random();
        int[] RBMLayerToCreate;

        public MLPwithoutTraining(BinaryForNetworkToLoad Binary)
        {
            for (int i = 0; i < Binary.AnzahlLayer; i++)
            {
                Test2.Add(new MLPCreateNeuralNetwork { Neurons = Binary.NeuronsPerLayer[i], ActivationFunction = Binary.AktivierungsFunktion[i], OutputFunction = Binary.OutputFunktion[i] });
            }
            //Test2.Add(new MLPCreateNeuralNetwork { Neurons = 784, ActivationFunction = new LineareFunktion(), OutputFunction = new LineareFunktion() });
            //Test2.Add(new MLPCreateNeuralNetwork { Neurons = 100, ActivationFunction = new TanhReLu(), OutputFunction = new LineareFunktion() });
            ////Test2.Add(new MLPCreateNeuralNetwork { Neurons = 200, ActivationFunction = new TangensHyperbolikusFunktion(), OutputFunction = new LineareFunktion() });
            //Test2.Add(new MLPCreateNeuralNetwork { Neurons = 10, ActivationFunction = new TanhReLu(), OutputFunction = new LineareFunktion() });
            //Test2.Add(new MLPCreateNeuralNetwork { Neurons = 3, ActivationFunction = new TanhReLu(), OutputFunction = new LineareFunktion() });
            //Test2.Add(new MLPCreateNeuralNetwork { Neurons = 3, ActivationFunction = new TangensHyperbolikusFunktion(), OutputFunction = new LineareFunktion() });
            RBMLayerToCreate = new int[Test2.Count];
            for (int i = 0; i < Test2.Count; i++)
            {
                RBMLayerToCreate[i] = Test2[i].Neurons;
            }

        }


        public void ErstellenErgebniss(string Matrixpfad, List<PatternToLearn> testSet, List<PatternToLearn> bildSet, string speicherpfad)
        {
            //string lernregel = "ERS"; //Eingabe ERS,ERS2 oder backprop
            double RBMLearningRate = 0.1;
            //int contrastiveDivergence = 100;
            //double MLPLearningRate = 0.1;
            //double MLPTolerance = 0.01;
            int temp = 0;


            foreach (var item in Test2)
            {
                temp += item.Neurons;
            }
            double[,] weightmatrix = new double[temp, temp];
            RBM2 = new RestrictedBoltzmannMachine2(RBMLayerToCreate, RND);

            weightmatrix = RBM2.Matrix.Clone(RBM2.Matrix, weightmatrix);
            TestNetz2 = new MultiLayerPerceptron2(Test2, Matrixpfad);


            string Time = DateTime.Now.ToString("dd.MM.yy_HHmmss");
            string Networksize = "";
            foreach (var Layer in Test2)
            {
                Networksize += Layer.Neurons + "_";
            }
            //int schritte = TestNetz2.Training(100000, MLPLearningRate, MLPTolerance, trainigsSet, RBM2.Bias, testSet, lernregel);
            double[] Vector = new double[testSet[0].inputvector.Length];
            double[] neu = new double[testSet[0].targetvector.Length];
            int numberOfCorrectClassification = 0;
            System.IO.TextWriter file = new System.IO.StreamWriter(speicherpfad + @"\" + Networksize + Time + "Genauigkeit" + ".csv", true);//, true

            foreach (var patter in testSet)
            {
                double alle = 0;
                Vector = patter.inputvector;
                neu = Vector.CalculateTargetWithBias(TestNetz2.Layers, TestNetz2.Matrix, RBM2.Bias, 1d);
                foreach (var n in neu)
                {
                    if (n < 0)
                    {
                        alle += 0;
                    }
                    else
                    {
                        alle += Math.Abs(n);
                    }
                }
                for (int i = 0; i < neu.Length; i++)
                {
                    if (neu[i] < 0)
                    {
                        neu[i] = 0;
                    }
                    else
                    {
                        neu[i] = Math.Abs(neu[i]) / alle;
                    }
                }
                if (patter.targetvector[0] == 1)
                {
                    if (neu[0] > neu[1] && neu[0] > neu[2])
                    {
                        numberOfCorrectClassification++;
                    }
                }
                if (patter.targetvector[1] == 1)
                {
                    if (neu[1] > neu[0] && neu[1] > neu[2])
                    {
                        numberOfCorrectClassification++;
                    }
                }
                if (patter.targetvector[2] == 1)
                {
                    if (neu[2] > neu[1] && neu[2] > neu[0])
                    {
                        numberOfCorrectClassification++;
                    }
                }
                for (int i = 0; i < neu.Length; i++)
                {
                    file.Write(neu[i] + ";" + patter.targetvector[i] + ";");
                    file.WriteLine();
                    file.Flush();

                    //using (System.IO.TextWriter file =
                    //new System.IO.StreamWriter(speicherpfad + @"\" + Networksize + Time + "Genauigkeit" + ".csv", true))//, true
                    //{
                    //    file.Write(neu[i] + ";" + patter.targetvector[i] + ";");
                    //    file.WriteLine();
                    //    file.Flush();
                    //    file.Close();
                    //}
                }
                file.WriteLine();
                file.Flush();
                //using (System.IO.TextWriter file =
                //new System.IO.StreamWriter(speicherpfad + @"\" + Networksize + Time + "Genauigkeit" + ".csv", true))//, true
                //{
                //    file.WriteLine();
                //    file.Flush();
                //    file.Close();
                //}
            }
            file.Write("Richtig Klassifizierte Muster" + ";" + numberOfCorrectClassification + ";");
            file.WriteLine();
            file.Flush();
            file.Close();
            //using (System.IO.TextWriter file =
            //        new System.IO.StreamWriter(speicherpfad + @"\" + Networksize + Time + "Genauigkeit" + ".csv", true))//, true
            //{
            //    file.Write("Richtig Klassifizierte Muster" + ";" + numberOfCorrectClassification + ";");
            //    file.WriteLine();
            //    file.Flush();
            //    file.Close();
            //}
            double[] TestVector = new double[784];
            for (int i = 0; i < 784; i++)
            {
                TestVector[i] = 0;
            }
            var bild = new Bilderstellen(28, 28);
            int counter = 0;
            for (int i = 0; i < 28; i++)
            {
                for (int j = 0; j < 28; j++)
                {
                    double alle = 0;
                    TestVector[counter] = 1;
                    neu = TestVector.CalculateTargetWithBias(TestNetz2.Layers, TestNetz2.Matrix, RBM2.Bias, 1d);
                    foreach (var n in neu)
                    {
                        if (n < 0)
                        {
                            alle += 0;
                        }
                        else
                        {
                            alle += Math.Abs(n);
                        }
                    }
                    for (int o = 0; o < neu.Length; o++)
                    {
                        if (neu[o] < 0)
                        {
                            neu[o] = 0;
                        }
                        else
                        {
                            neu[o] = Math.Abs(neu[o]) / alle;
                        }
                    }
                    if (neu[0] > neu[1] && neu[0] > neu[2]) // ist keine Spirale Farbe Weiß
                    {
                        bild.BildPixelPlazieren(i, j, System.Drawing.Color.White);
                    }


                    if (neu[1] > neu[0] && neu[1] > neu[2]) // ist Spirale 1 Farbe Blau
                    {
                        bild.BildPixelPlazieren(i, j, System.Drawing.Color.Blue);
                    }


                    if (neu[2] > neu[1] && neu[2] > neu[0]) // ist Spirale 2 Farbe Rot
                    {
                        bild.BildPixelPlazieren(i, j, System.Drawing.Color.Red);
                    }
                    TestVector[counter] = 0;
                    counter++;

                }
                bild.SavePicture(speicherpfad + @"\" + Networksize + Time + "Bild" + ".png");
                Vector = bildSet[0].inputvector;
                //RBM2.GreedyTrainingRestrictedBoltzmannMachineWithPicture(Vector, RBMLearningRate, RND);
            }

        }

    }
}
