using DeepBeliefNeuralNetwork.MLPComponents;
using DeepBeliefNeuralNetwork.MLPComponents.Funktionen;
using DeepBeliefNeuralNetwork.RBMComponents;
using System;
using System.Collections.Generic;

namespace DeepBeliefNeuralNetwork
{
    public class DeepBeliefNetwork
    {
        private List<RestrictedBoltzmannMachine> RBMLayer = new List<RestrictedBoltzmannMachine>();
        private List<MLPCreateNeuralNetwork> Test = new List<MLPCreateNeuralNetwork>();
        private List<MLPCreateNeuralNetwork> KNN = new List<MLPCreateNeuralNetwork>();
        private RestrictedBoltzmannMachine2 RBM2;
        private MultilayerPerceptron TestNetz;
        private MultiLayerPerceptron2 TestNetz2;

        //private Random RND = new Random();
        private ThreadSafeRandom RND = new ThreadSafeRandom();

        private int[] RBMLayerToCreate;

        //Auswahl mit welcher Methode gelernt werden soll
        private bool RBM_MLP = true;//Variante 1

        private bool MLP = false;// Variante 2
        private bool DBNN = false;// Variante 3

        private string _lernregel;

        public DeepBeliefNetwork(string lernregel)
        {
            _lernregel = lernregel;

            KNN.Add(new MLPCreateNeuralNetwork { Neurons = 2304, ActivationFunction = new LineareFunktion(), OutputFunction = new LineareFunktion() });
            //KNN.Add(new MLPCreateNeuralNetwork { Neurons = 1000, ActivationFunction = new ReLu(), OutputFunction = new LineareFunktion() });
            //KNN.Add(new MLPCreateNeuralNetwork { Neurons = 100, ActivationFunction = new ReLu(), OutputFunction = new LineareFunktion() });
            KNN.Add(new MLPCreateNeuralNetwork { Neurons = 43, ActivationFunction = new ReLu(), OutputFunction = new LineareFunktion() });
            
            /*
            KNN.Add(new MLPCreateNeuralNetwork { Neurons = 2, ActivationFunction = new LineareFunktion(), OutputFunction = new LineareFunktion() });
            KNN.Add(new MLPCreateNeuralNetwork { Neurons = 2, ActivationFunction = new SigmoideFunktion(), OutputFunction = new LineareFunktion() });
            KNN.Add(new MLPCreateNeuralNetwork { Neurons = 1, ActivationFunction = new LineareFunktion(), OutputFunction = new LineareFunktion() });
            */
            RBMLayerToCreate = new int[KNN.Count];
            for (int i = 0; i < KNN.Count; i++)
            {
                RBMLayerToCreate[i] = KNN[i].Neurons;
            }
        }

        public void GreedyLayerWiseTraining(List<PatternToLearn> trainigsSet, List<PatternToLearn> testSet, List<PatternToLearn> bildSet, string Variance)
        {
            string desktop = Environment.GetFolderPath(Environment.SpecialFolder.Desktop);
            string OrdnerBachelorarbeit = "BachelorarbeitJoseph";
            string Ordnerpfad = desktop + @"\" + OrdnerBachelorarbeit + @"\Berechnung" + "BerechnungVariante2ERS" + @"\";

            #region RBM_MLP

            if (RBM_MLP)
            {
                //string lernregel = "backprop"; //Eingabe ERS,ERS2 oder backprop
                double RBMLearningRate = 0.1;
                int contrastiveDivergence = 10000;
                double MLPLearningRate = 0.1;
                double MLPTolerance = 0.01;
                int temp = 0, counter = 1;

                //Ordnerpfad = desktop + @"\" + OrdnerBachelorarbeit + @"\Berechnung\Mean0Variance2" + @"\" + @"CD" + contrastiveDivergence + @"\";
                Ordnerpfad = desktop;

                foreach (var item in KNN)
                {
                    temp += item.Neurons;
                }
                double[,] weightmatrix = new double[temp, temp];
                RBM2 = new RestrictedBoltzmannMachine2(RBMLayerToCreate, RND);
                var myBag = new System.Collections.Concurrent.ConcurrentBag<PatternToLearn>(trainigsSet);
                System.Threading.Tasks.Parallel.ForEach(myBag, new System.Threading.Tasks.ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount }, trainingCase =>
                {
                    var inputVector = (double[])trainingCase.inputvector.Clone();
                    RBM2.GreedyTrainingRestrictedBoltzmannMachine(inputVector, contrastiveDivergence, RBMLearningRate, RND);
                    Console.WriteLine(counter++ + "/" + myBag.Count);
                });
                /*
                foreach (var trainingCase in trainigsSet)
                {
                    var inputVector = (double[])trainingCase.inputvector.Clone();
                    RBM2.GreedyTrainingRestrictedBoltzmannMachine(inputVector, contrastiveDivergence, RBMLearningRate, RND);
                }
                */
                weightmatrix = RBM2.Matrix.Clone(RBM2.Matrix, weightmatrix);
                TestNetz2 = new MultiLayerPerceptron2(KNN, weightmatrix, RND);

                string Time = DateTime.Now.ToString("dd.MM.yy_HHmmss");
                string Networksize = "";
                foreach (var Layer in KNN)
                {
                    Networksize += Layer.Neurons + "_";
                }
                /*
                System.IO.TextWriter file = new System.IO.StreamWriter(Ordnerpfad + Networksize + Time + "Genauigkeit" + ".csv", true);//, true

                file.Write(KNN[0].ActivationFunction.ToString() + ";" + KNN[1].ActivationFunction.ToString() + ";" + KNN[2].ActivationFunction.ToString() + ";" + KNN[3].ActivationFunction.ToString());
                file.WriteLine();
                file.Write("RBM Lernrate" + ";" + RBMLearningRate + ";" + "Contrastive Divergence" + ";" + contrastiveDivergence + ";" + "MLP Lernrate" + ";" + MLPLearningRate + ";" + "MLP Toleranz" + ";" + MLPTolerance);
                file.WriteLine();
                file.Write("Lernregel" + ";" + _lernregel + ";");
                file.WriteLine();
                file.Flush();
                */
                int schritte = TestNetz2.Training(10000, MLPLearningRate, MLPTolerance, trainigsSet, RBM2.Bias, testSet, _lernregel, Ordnerpfad);
                double[] Vector = new double[trainigsSet[0].inputvector.Length];
                double[] neu = new double[trainigsSet[0].targetvector.Length];
                int numberOfCorrectClassification = 0;
                /*
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
                            numberOfCorrectClassification += 1;
                        }
                    }
                    if (patter.targetvector[1] == 1)
                    {
                        if (neu[1] > neu[0] && neu[1] > neu[2])
                        {
                            numberOfCorrectClassification += 1;
                        }
                    }
                    if (patter.targetvector[2] == 1)
                    {
                        if (neu[2] > neu[1] && neu[2] > neu[0])
                        {
                            numberOfCorrectClassification += 1;
                        }
                    }
                    for (int i = 0; i < neu.Length; i++)

                    {
                        file.Write(neu[i] + ";" + patter.targetvector[i] + ";");
                        file.WriteLine();
                        file.Flush();
                    }
                    file.WriteLine();
                    file.Flush();
                }
                file.Write("Richtig Klassifizierte Muster" + ";" + numberOfCorrectClassification + ";");
                file.WriteLine();
                file.Flush();
                file.Close();

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
                }
                bild.SavePicture(Ordnerpfad + Networksize + Time + "Bild" + ".png");
                */
                //Vector = bildSet[0].inputvector;
                //RBM2.GreedyTrainingRestrictedBoltzmannMachineWithPicture(Vector, RBMLearningRate, RND);
            }

            #endregion RBM_MLP

            #region DBNN

            if (DBNN)
            {
                double[] inputVector;
                double RBMLearningRate = 0.1;
                int contrastiveDivergence = 10000;
                double MLPLearningRate = 0.5;
                double MLPTolerance = 0.01;
                int RBM1Visible = 784;
                int RBM1Hidden = 200;
                int RBM2Visible = 200;
                int RBM2Hidden = 50;
                RBMLayer.Add(new RestrictedBoltzmannMachine(RBM1Visible, RBM1Hidden, RBMLearningRate, RND));
                RBMLayer.Add(new RestrictedBoltzmannMachine(RBM2Visible, RBM2Hidden, RBMLearningRate, RND));
                TestNetz = new MultilayerPerceptron(Test);

                for (int i = 0; i < 1; i++)
                {
                    foreach (var trainingCase in trainigsSet)
                    {
                        inputVector = (double[])trainingCase.inputvector.Clone();
                        foreach (var RBM in RBMLayer)//Layer wise greedy learning für die andere Variante
                        {
                            inputVector = RBM.GreedyLearning(inputVector, contrastiveDivergence, RND);
                        }
                    }
                }

                foreach (var trainingCase in trainigsSet)
                {
                    inputVector = (double[])trainingCase.inputvector.Clone();
                    foreach (var RBM in RBMLayer)
                    {
                        double[] outputvector = new double[RBM.HiddenNeurons.Count];
                        for (int i = 0; i < RBM.VisualNeurons.Count; i++)
                        {
                            RBM.VisualNeurons[i].BinaryState = inputVector[i];
                        }
                        RBMContrastiveDivergence training = new RBMContrastiveDivergence();
                        foreach (var hiddenNeuron in RBM.HiddenNeurons)
                        {
                            hiddenNeuron.BinaryState = training.CalculateHiddenBinaryState(RBM.BiasNeuron.BiasHidden,
                                hiddenNeuron, RBM.WeightMatrix, RBM.VisualNeurons, RBM.NumberOfVisualNeurons, RND);
                        }
                        for (int i = 0; i < RBM.HiddenNeurons.Count; i++)
                        {
                            outputvector[i] = RBM.HiddenNeurons[i].Probability;
                        }
                        inputVector = (double[])outputvector.Clone();
                        //inputVector = RBM.CalculateOutput(inputVector,RBM, RND);
                    }
                    trainingCase.RBMOutput = (double[])inputVector.Clone();
                }

                string Time = DateTime.Now.ToString("dd.MM.yy_HHmmss");
                string Networksize = "";
                foreach (var Layer in Test)
                {
                    Networksize += Layer.Neurons + "_";
                }
                using (System.IO.TextWriter file =
                new System.IO.StreamWriter(@"C:\Users\Joseph\Desktop\" + Networksize + Time + ".csv", true))//, true
                {
                    file.Write(Test[0].ActivationFunction.ToString() + ";" + Test[1].ActivationFunction.ToString() + ";" + Test[2].ActivationFunction.ToString() + ";");
                    //file.Write(Test[0].ActivationFunction.ToString() + ";" + Test[1].ActivationFunction.ToString() + ";"+ Test[2].ActivationFunction.ToString() + ";"+ Test[3].ActivationFunction.ToString() + ";");
                    file.WriteLine();
                    file.Write("RBM Lernrate" + ";" + RBMLearningRate + ";" + "Contrastive Divergence" + ";" + contrastiveDivergence + ";" + "MLP Lernrate" + ";" + MLPLearningRate + ";" + "MLP Toleranz" + ";" + MLPTolerance);
                    file.WriteLine();
                    file.Write("Lernregel" + ";" + "backprop" + ";" + "RBMs" + ";" + RBMLayer.Count + ";" + "RBM1" + ";" + RBM1Visible + "-" + RBM1Hidden + ";" + "RBM2" + ";" + RBM2Visible + "-" + RBM2Hidden + ";");
                    //file.Write("Lernregel" + ";" + "ERS" + ";" + "RBMs" + ";" + RBMLayer.Count + ";" + "RBM1" + ";" +  RBM1Visible + "-" + RBM1Hidden + ";" + "RBM2" + ";");
                    file.WriteLine();
                    file.Flush();
                }
                int schritte = TestNetz.Training(100000, MLPLearningRate, MLPTolerance, trainigsSet, testSet, RBMLayer, RND);

                double[] Vector = new double[trainigsSet[0].inputvector.Length];
                double[] neu = new double[trainigsSet[0].targetvector.Length];
                int numberOfCorrectClassification = 0;

                foreach (var patter in testSet)
                {
                    inputVector = (double[])patter.inputvector.Clone();
                    foreach (var RBM in RBMLayer)
                    {
                        double[] outputvector = new double[RBM.HiddenNeurons.Count];
                        for (int i = 0; i < RBM.VisualNeurons.Count; i++)
                        {
                            RBM.VisualNeurons[i].BinaryState = inputVector[i];
                        }
                        RBMContrastiveDivergence training = new RBMContrastiveDivergence();
                        foreach (var hiddenNeuron in RBM.HiddenNeurons)
                        {
                            hiddenNeuron.BinaryState = training.CalculateHiddenBinaryState(RBM.BiasNeuron.BiasHidden,
                                hiddenNeuron, RBM.WeightMatrix, RBM.VisualNeurons, RBM.NumberOfVisualNeurons, RND);
                        }
                        for (int i = 0; i < RBM.HiddenNeurons.Count; i++)
                        {
                            outputvector[i] = RBM.HiddenNeurons[i].Probability;
                        }
                        inputVector = (double[])outputvector.Clone();
                        //inputVector = RBM.CalculateOutput(inputVector,RBM, RND);
                    }
                    patter.RBMOutput = (double[])inputVector.Clone();

                    double alle = 0;
                    Vector = patter.RBMOutput;
                    neu = Vector.CalculateTarget(TestNetz.Layers, TestNetz.Matrix, 1d);
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
                            numberOfCorrectClassification += 1;
                        }
                    }
                    if (patter.targetvector[1] == 1)
                    {
                        if (neu[1] > neu[0] && neu[1] > neu[2])
                        {
                            numberOfCorrectClassification += 1;
                        }
                    }
                    if (patter.targetvector[2] == 1)
                    {
                        if (neu[2] > neu[1] && neu[2] > neu[0])
                        {
                            numberOfCorrectClassification += 1;
                        }
                    }
                    for (int i = 0; i < neu.Length; i++)

                    {
                        using (System.IO.TextWriter file =
                    new System.IO.StreamWriter(@"C:\Users\Joseph\Desktop\" + Networksize + Time + "Genauigkeit" + ".csv", true))//, true
                        {
                            file.Write(neu[i] + ";" + patter.targetvector[i] + ";");
                            file.WriteLine();
                            file.Flush();
                        }
                    }
                    using (System.IO.TextWriter file =
                    new System.IO.StreamWriter(@"C:\Users\Joseph\Desktop\" + Networksize + Time + "Genauigkeit" + ".csv", true))//, true
                    {
                        file.WriteLine();
                        file.Flush();
                    }
                }
                using (System.IO.TextWriter file =
                        new System.IO.StreamWriter(@"C:\Users\Joseph\Desktop\" + Networksize + Time + "Genauigkeit" + ".csv", true))//, true
                {
                    file.Write("Richtig Klassifizierte Muster" + ";" + numberOfCorrectClassification + ";");
                    file.WriteLine();
                    file.Flush();
                }
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

                        inputVector = (double[])TestVector.Clone();
                        foreach (var RBM in RBMLayer)
                        {
                            double[] outputvector = new double[RBM.HiddenNeurons.Count];
                            for (int k = 0; k < RBM.VisualNeurons.Count; k++)
                            {
                                RBM.VisualNeurons[k].BinaryState = inputVector[k];
                            }
                            RBMContrastiveDivergence training = new RBMContrastiveDivergence();
                            foreach (var hiddenNeuron in RBM.HiddenNeurons)
                            {
                                hiddenNeuron.BinaryState = training.CalculateHiddenBinaryState(RBM.BiasNeuron.BiasHidden,
                                    hiddenNeuron, RBM.WeightMatrix, RBM.VisualNeurons, RBM.NumberOfVisualNeurons, RND);
                            }
                            for (int k = 0; k < RBM.HiddenNeurons.Count; k++)
                            {
                                outputvector[k] = RBM.HiddenNeurons[k].Probability;
                            }
                            inputVector = (double[])outputvector.Clone();
                            //inputVector = RBM.CalculateOutput(inputVector,RBM, RND);
                        }

                        neu = inputVector.CalculateTarget(TestNetz.Layers, TestNetz.Matrix, 1d);
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
                }
                bild.SavePicture(@"C:\Users\Joseph\Desktop\" + Networksize + Time + "Bild" + ".png");
            }

            #endregion DBNN

            #region MLP

            if (MLP)
            {
                string lernregel = "ERS"; //Eingabe ERS,ERS2 oder backprop
                double MLPLearningRate = 0.1;
                double MLPTolerance = 0.01;
                //Ordnerpfad = Desktop + @"\" + OrdnerBachelorarbeit + @"\Berechnung\" + Variance + @"\";

                RBM2 = new RestrictedBoltzmannMachine2(RBMLayerToCreate, RND);

                TestNetz2 = new MultiLayerPerceptron2(KNN, null, RND);

                string Time = DateTime.Now.ToString("dd.MM.yy_HHmmss");
                string Networksize = "";
                foreach (var Layer in KNN)
                {
                    Networksize += Layer.Neurons + "_";
                }
                System.IO.TextWriter file = new System.IO.StreamWriter(Ordnerpfad + Networksize + Time + "Genauigkeit" + ".csv", true);//, true

                file.Write(KNN[0].ActivationFunction.ToString() + ";" + KNN[1].ActivationFunction.ToString() + ";" + KNN[2].ActivationFunction.ToString() + ";" + KNN[3].ActivationFunction.ToString());
                file.WriteLine();
                file.Write(";" + "MLP Lernrate" + ";" + MLPLearningRate + ";" + "MLP Toleranz" + ";" + MLPTolerance);
                file.WriteLine();
                file.Write("Lernregel" + ";" + lernregel + ";");
                file.WriteLine();
                file.Flush();

                int schritte = TestNetz2.Training(100000, MLPLearningRate, MLPTolerance, trainigsSet, RBM2.Bias, testSet, lernregel, Ordnerpfad);
                //Console.WriteLine("Notwendige Trainingsschritte:" + " " + schritte);
                //Console.WriteLine("Notwendige Trainingsschritte:" + " " + schritte);
                double[] Vector = new double[trainigsSet[0].inputvector.Length];
                double[] neu = new double[trainigsSet[0].targetvector.Length];
                int numberOfCorrectClassification = 0;

                foreach (var patter in testSet)
                {
                    double alle = 0;
                    Vector = patter.inputvector;
                    neu = Vector.CalculateTarget(TestNetz2.Layers, TestNetz2.Matrix, 1d);
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
                            numberOfCorrectClassification += 1;
                        }
                    }
                    if (patter.targetvector[1] == 1)
                    {
                        if (neu[1] > neu[0] && neu[1] > neu[2])
                        {
                            numberOfCorrectClassification += 1;
                        }
                    }
                    if (patter.targetvector[2] == 1)
                    {
                        if (neu[2] > neu[1] && neu[2] > neu[0])
                        {
                            numberOfCorrectClassification += 1;
                        }
                    }
                    for (int i = 0; i < neu.Length; i++)
                    {
                        file.Write(neu[i] + ";" + patter.targetvector[i] + ";");
                        file.WriteLine();
                        file.Flush();
                    }

                    file.WriteLine();
                    file.Flush();
                }

                file.Write("Richtig Klassifizierte Muster" + ";" + numberOfCorrectClassification + ";");
                file.WriteLine();
                file.Flush();
                file.Close();
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
                }
                bild.SavePicture(Ordnerpfad + Networksize + Time + "Bild" + ".png");
            }

            #endregion MLP
        }
    }
}