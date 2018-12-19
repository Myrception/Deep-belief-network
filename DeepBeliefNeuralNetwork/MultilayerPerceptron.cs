using System;
using System.Collections.Generic;
using DeepBeliefNeuralNetwork.MLPComponents;
using DeepBeliefNeuralNetwork.MLPComponents.Funktionen;
using DeepBeliefNeuralNetwork.RBMComponents;

namespace DeepBeliefNeuralNetwork
{
    internal class MultilayerPerceptron
    {
        public decimal Fehler = 0;
        public List<List<MLPNeuron>> Layers = new List<List<MLPNeuron>>();
        public MLPWeightMatrix Matrix { get; set; }

        private double[,] CloneMatrix;

        #region Constructor

        /// <summary>
        /// Im Konstruktor wird die Gewichtsmatrix mit der nötigen Größe initialisiert. Sowie die
        /// Eingabeneuronen und Ausgabeneuronen erstellt. Danach wird die Matrix noch an den Verbindungsstellen
        /// mit Random werten belegt.
        /// </summary>
        /// <param name="networkToBeCreate">Enthält alle Parameter die zur erstellung des Neuronalen Netzes notwendig sind</param>
        public MultilayerPerceptron(List<MLPCreateNeuralNetwork> networkToBeCreate)
        {
            int index = 0, count = 0;
            foreach (MLPCreateNeuralNetwork item in networkToBeCreate)
            {
                count += item.Neurons;
                List<MLPNeuron> temp = new List<MLPNeuron>();
                for (int i = 0; i < item.Neurons; i++)
                {
                    temp.Add(new MLPNeuron() { Index = index, ActivationFunction = item.ActivationFunction, OutputFunction = item.OutputFunction });
                    index++;
                }
                Layers.Add(temp);
            }
            Matrix = new MLPWeightMatrix(count, count);
            CloneMatrix = new double[count, count];
            MatrixRandom();
        }

        #endregion Constructor

        #region Neuronal Network training

        /// <summary>
        /// In dieser Methode wird das Neuronale Netz trainiert.
        /// </summary>
        /// <param name="maxLearningSteps">Gibt an wieviele Lernschritte maximal gemacht werden sollen</param>
        /// <param name="learningRate">Legt die Lernrate der Lernregel fest</param>
        /// <param name="learningToleranz">Legt dei Toleranz mit der gelernt werden soll fest</param>
        /// <param name="patternToLearn">Enthalten die Trainingsmuster</param>
        /// <returns></returns>
        internal int Training(int maxLearningSteps, double learningRate, double learningToleranz, List<PatternToLearn> patternToLearn, List<PatternToLearn> patternToTest, List<RestrictedBoltzmannMachine> RBMLayer,Random RND)
        {
            string time = DateTime.Now.ToString("dd.MM.yy_HHmmss");
            string networksize = "";
            bool trainingErfolgt = false;
            int steps = 0;
            foreach (var Layer in Layers)
            {
                networksize += Layer.Count + "_";
            }
            while (steps < maxLearningSteps)
            {

                Fehler = 0;
                decimal fehlerTest = 0;
                foreach (var muster in patternToLearn)
                {
                    var inputVector = (double[])muster.inputvector.Clone();
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
                    muster.RBMOutput = (double[])inputVector.Clone();
                    if (TrainiereMuster(learningRate, learningToleranz, muster))
                    {
                        trainingErfolgt = true;
                    }
                    //if (TrainiereMusterERS(learningRate, learningToleranz, muster))
                    //{
                    //    trainingErfolgt = true;
                    //}
                }
                foreach (var patter in patternToTest)
                {
                    var inputVector = (double[])patter.inputvector.Clone();
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


                    double[] pruefung = new double[patter.targetvector.Length];
                    double[] ausgabevektor = new double[patter.targetvector.Length];
                    ausgabevektor = patter.RBMOutput.CalculateTarget(Layers, Matrix, 1d);
                    for (int i = 0; i < patter.targetvector.Length; i++)
                    {
                        pruefung[i] = patter.targetvector[i] - ausgabevektor[i];
                    }
                    foreach (double t in pruefung)
                    {
                        double toleranz = learningToleranz - Math.Abs(t);
                        fehlerTest += (decimal)Math.Abs(t);
                    }
                }
                using (System.IO.TextWriter file =
                new System.IO.StreamWriter(@"C:\Users\Joseph\Desktop\" + networksize + time + ".csv", true))//, true
                {
                    file.Write(steps + ";" + Fehler.ToString() + ";" + fehlerTest.ToString() + ";");
                    file.WriteLine();
                    file.Flush();
                }
                Console.WriteLine("Trainingserror:" + " " + Fehler);
                Console.WriteLine("Testerror:" + " " + fehlerTest);
                
                if (trainingErfolgt)
                {
                    Console.WriteLine(steps);
                    steps++;
                    trainingErfolgt = false;
                }
                else
                {
                    Console.WriteLine(Fehler);
                    
                    
                        Matrix.Speichern(@"C:\Users\Joseph\Desktop\Matrix.csv", Matrix);
                        return steps;
                    
                }
            }
            //if (steps == maxLearningSteps)
            //{
            //    for (int i = 0; i < Matrix.Spalten; i++)
            //    {
            //        for (int j = 0; j < Matrix.Zeilen; j++)
            //        {
            //            Matrix[i, j] = CloneMatrix[i, j];
            //        }
            //    }
            //}
            Matrix.Speichern(@"C:\Users\Joseph\Desktop\Matrix.csv", Matrix);
            return steps;
        }

        /// <summary>
        /// Hier findet das eigentliche Training statt. Implementiert ist die Backpropagation Lernregel. Ja sieht komisch aus funktioniert aber.
        /// </summary>
        /// <param name="lernrate">Legt die Lernrate fest</param>
        /// <param name="lerntoleranz">Legt die Toleranz fest mit der gelernt wird</param>
        /// <param name="muster">Die Muster die gelernt werden sollen</param>
        /// <returns></returns>
        private bool TrainiereMuster(double lernrate, double lerntoleranz, PatternToLearn muster)
        {
            bool nochmaltrainig = false;
            double[] pruefung = new double[muster.targetvector.Length];
            double[] ausgabevektor = new double[muster.targetvector.Length];
            ausgabevektor = muster.RBMOutput.CalculateTarget(Layers, Matrix, 1d);
            List<MLPBackpropagationDelta> backDelta = new List<MLPBackpropagationDelta>(); List<MLPWeightChange> änderungen = new List<MLPWeightChange>();
            for (int i = 0; i < Layers[Layers.Count - 1].Count; i++)
            {
                pruefung[i] = muster.targetvector[i] - ausgabevektor[i];
               
                if (double.IsNaN(pruefung[i]))
                {
                    
                }
                double outputDelta;
                if (Layers[Layers.Count - 1][i].ActivationFunction is Softmax)
                {
                   outputDelta = Layers[Layers.Count - 1][i].ActivationFunction.BerechneAbleitung(Layers[Layers.Count - 1][i].NetInput, 1d,Layers)
                     * (muster.targetvector[i] - Layers[Layers.Count - 1][i].Output);
                    if (double.IsNaN(outputDelta)||double.IsInfinity(outputDelta))
                    {
                        nochmaltrainig = false;
                    }
                }
                else
                {    
                outputDelta = Layers[Layers.Count - 1][i].ActivationFunction.BerechneAbleitung(Layers[Layers.Count - 1][i].NetInput, 1d)
                    * (muster.targetvector[i] - Layers[Layers.Count - 1][i].Output);
                }
                if (double.IsNaN(outputDelta))
                {
                    backDelta.Add(new MLPBackpropagationDelta() { Index = Layers[Layers.Count - 1][i].Index, Delta = 0 });
                }
                else
                {
                    foreach (MLPNeuron neuronenOutput in Layers[Layers.Count - 2])
                    {
                        var änderung = new MLPWeightChange
                        {
                            Zeile = neuronenOutput.Index,
                            Spalte = Layers[Layers.Count - 1][i].Index,
                            Änderungswert = lernrate * outputDelta * neuronenOutput.Output
                        };
                        änderungen.Add(änderung);
                    }
                    backDelta.Add(new MLPBackpropagationDelta() { Index = Layers[Layers.Count - 1][i].Index, Delta = outputDelta });
                }
            }
            for (int i = Layers.Count - 2; i > 0; i--)//Schicht der Schichten
            {
                for (int j = Layers[i][Layers[i].Count - 1].Index; j >= Layers[i][0].Index; j--)//Auf Neuronen Ebene
                {
                    double temp = 0;
                    foreach (MLPNeuron item in Layers[i + 1])
                    {
                        foreach (MLPBackpropagationDelta item2 in backDelta)
                        {
                            if (item.Index == item2.Index)
                            {
                                if (double.IsNaN(item2.Delta))
                                {
                                    temp += 0;
                                }
                                else if (double.IsNaN(Matrix[j, item2.Index]))
                                {
                                    Matrix[j, item2.Index] = double.NaN;
                                }
                                else
                                {
                                    temp += item2.Delta * Matrix[j, item2.Index];
                                }
                            }
                        }
                    }
                    if (Layers[i][j - Layers[i][0].Index].ActivationFunction is Softmax)
                    {
                        throw new NotImplementedException();
                    }
                    else
                    {
                        
                    temp *= Layers[i][j - Layers[i][0].Index].ActivationFunction.BerechneAbleitung(Layers[i][j - Layers[i][0].Index].NetInput, 1d);
                    }
                    foreach (MLPNeuron neuronenOutput in Layers[i - 1])
                    {
                        var änderung = new MLPWeightChange
                        {
                            Zeile = neuronenOutput.Index,
                            Spalte = j,
                            Änderungswert = lernrate * temp * neuronenOutput.Output
                        };
                        änderungen.Add(änderung);
                    }
                    backDelta.Add(new MLPBackpropagationDelta() { Index = j, Delta = temp });
                }
            }

            foreach (double t in pruefung)
            {
                double toleranz = lerntoleranz - Math.Abs(t);
                if (toleranz < 0)
                {
                    nochmaltrainig = true;
                }
                Fehler += (decimal)Math.Abs(t);
            }
            if (nochmaltrainig)
            {
                foreach (var item in änderungen)
                {
                    if (!double.IsNaN(Matrix[item.Zeile, item.Spalte]))
                    {
                        Matrix[item.Zeile, item.Spalte] += item.Änderungswert;
                    }
                }
            }
            return nochmaltrainig;
        }

        #region ERS Training
        private bool TrainiereMusterERS(double lernrate, double lerntoleranz, PatternToLearn muster)
        {
            bool nochmaltrainig = false;
            double[] pruefung = new double[muster.targetvector.Length];
            double[] ausgabevektor = new double[muster.targetvector.Length];
            ausgabevektor = muster.RBMOutput.CalculateTarget(Layers, Matrix, 1d);
            List<MLPBackpropagationDelta> backDelta = new List<MLPBackpropagationDelta>();
            List<MLPWeightChange> änderungen = new List<MLPWeightChange>();
            for (int i = 0; i < Layers[Layers.Count - 1].Count; i++)
            {
                pruefung[i] = muster.targetvector[i] - ausgabevektor[i];// Kosten Funktion
                //pruefung[i] = -(muster.targetvector[i]) * Math.Log(ausgabevektor[i]); // Kosten Funktion für Softmax

                if (double.IsNaN(pruefung[i]))
                {

                }
                double outputDelta;
                //if (Layers[Layers.Count - 1][i].ActivationFunction is Softmax)
                //{
                //    outputDelta = Layers[Layers.Count - 1][i].ActivationFunction.BerechneAbleitung(Layers[Layers.Count - 1][i].NetInput, 1d, Layers)
                //      * (muster.targetvector[i] - Layers[Layers.Count - 1][i].Output);
                //    if (double.IsNaN(outputDelta) || double.IsInfinity(outputDelta))
                //    {
                //        nochmaltrainig = false;
                //    }
                //}

                outputDelta = (muster.targetvector[i] - Layers[Layers.Count - 1][i].Output);

                //Layers[Layers.Count - 1][i].ActivationFunction.BerechneAbleitung(Layers[Layers.Count - 1][i].NetInput, 1d)
                //* (muster.targetvector[i] - Layers[Layers.Count - 1][i].Output);

                if (double.IsNaN(outputDelta))
                {
                    backDelta.Add(new MLPBackpropagationDelta() { Index = Layers[Layers.Count - 1][i].Index, Delta = 0 });
                }
                else
                {
                    foreach (MLPNeuron neuronenOutput in Layers[Layers.Count - 2])
                    {
                        var änderung = new MLPWeightChange
                        {
                            Zeile = neuronenOutput.Index,
                            Spalte = Layers[Layers.Count - 1][i].Index,
                            Änderungswert = lernrate * Math.Abs(1 - Math.Abs(Matrix[neuronenOutput.Index, Layers[Layers.Count - 1][i].Index])) * outputDelta * Math.Sign(neuronenOutput.Output)
                        };
                        änderungen.Add(änderung);
                        if (lernrate * Math.Abs(1 - Math.Abs(Matrix[neuronenOutput.Index, Layers[Layers.Count - 1][i].Index])) * outputDelta * Math.Sign(neuronenOutput.Output) > 100d)
                        {

                        }
                    }
                    backDelta.Add(new MLPBackpropagationDelta() { Index = Layers[Layers.Count - 1][i].Index, Delta = outputDelta });
                }
            }
            for (int i = Layers.Count - 2; i > 0; i--)//Schicht der Schichten
            {
                for (int j = Layers[i][Layers[i].Count - 1].Index; j >= Layers[i][0].Index; j--)//Auf Neuronen Ebene
                {
                    double temp = 0;
                    foreach (MLPNeuron item in Layers[i + 1])
                    {
                        foreach (MLPBackpropagationDelta item2 in backDelta)
                        {
                            if (item.Index == item2.Index)
                            {
                                if (double.IsNaN(item2.Delta))
                                {
                                    temp += 0;
                                }
                                else if (double.IsNaN(Matrix[j, item2.Index]))
                                {
                                    Matrix[j, item2.Index] = double.NaN;
                                }
                                else
                                {
                                    temp += item2.Delta * Matrix[j, item2.Index];
                                }
                            }
                        }
                    }
                    //if (Layers[i][j - Layers[i][0].Index].ActivationFunction is Softmax)
                    //{
                    //    throw new NotImplementedException();
                    //}
                    //else
                    //{

                    //    temp = temp * Layers[i][j - Layers[i][0].Index].ActivationFunction.BerechneAbleitung(Layers[i][j - Layers[i][0].Index].NetInput, 1d);
                    //}
                    foreach (MLPNeuron neuronenOutput in Layers[i - 1])
                    {
                        var änderung = new MLPWeightChange
                        {
                            Zeile = neuronenOutput.Index,
                            Spalte = j,
                            Änderungswert = lernrate * Math.Abs(1 - Math.Abs(Matrix[neuronenOutput.Index, j])) * temp * Math.Sign(neuronenOutput.Output)
                        };
                        änderungen.Add(änderung);
                        if (lernrate * Math.Abs(1 - Math.Abs(Matrix[neuronenOutput.Index, j])) * temp * Math.Sign(neuronenOutput.Output) > 100d)
                        {

                        }
                    }
                    backDelta.Add(new MLPBackpropagationDelta() { Index = j, Delta = temp });
                }
            }

            foreach (double t in pruefung)
            {
                double toleranz = lerntoleranz - Math.Abs(t);
                if (toleranz < 0)
                {
                    nochmaltrainig = true;
                }
                Fehler += (decimal)Math.Abs(t);
            }
            if (nochmaltrainig)
            {
                foreach (var item in änderungen)
                {
                    if (Matrix[item.Zeile, item.Spalte] + item.Änderungswert > 1d)
                    {
                        Matrix[item.Zeile, item.Spalte] = 1d;
                    }
                    else if (Matrix[item.Zeile, item.Spalte] + item.Änderungswert < -1d)
                    {
                        Matrix[item.Zeile, item.Spalte] = -1d;
                    }
                    else if (double.IsNaN(item.Änderungswert))
                    {
                        Matrix[item.Zeile, item.Spalte] = 1d;
                    }
                    else if (double.IsInfinity(item.Änderungswert))
                    {
                        Matrix[item.Zeile, item.Spalte] = 1d;
                    }
                    else
                    {
                        Matrix[item.Zeile, item.Spalte] += item.Änderungswert;
                    }
                }
            }
            if (nochmaltrainig == false)
            {

            }
            return nochmaltrainig;
        }
        #endregion

        #endregion Neuronal Network training

        private void MatrixRandom()
        {
            Random rnd = new Random();
            for (int i = 0; i < Layers.Count - 1; i++)
            {
                foreach (MLPNeuron layerN in Layers[i])
                {
                    foreach (MLPNeuron layerNPlusOne in Layers[i + 1])
                    {
                        Matrix[layerN.Index, layerNPlusOne.Index] = 2*(rnd.NextDouble()-0.5);
                    }
                }
            }
        }
    }
}
