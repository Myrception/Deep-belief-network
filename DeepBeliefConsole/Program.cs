using DeepBeliefNeuralNetwork;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;

namespace DeepBeliefConsole
{
    internal class Program
    {
        /// <summary>
        /// Großteil aller einstellungen sind in der Klasse DeepBeliefNetwork zu finden.
        /// </summary>
        /// <param name="args"></param>
        private static void Main(string[] args)
        {
            string Desktop = Environment.GetFolderPath(Environment.SpecialFolder.Desktop);
            string Trainingspfad = Desktop + @"\" + @"TraficSignPictures\train\GTSRB\Final_Training\Images";
            string Testpfad = Desktop + @"\" + @"TraficSignPictures\test\GTSRB\Final_Test\Images";

            string Variance = Testpfad.Split(@"\\".ToCharArray())[Testpfad.Split(@"\\".ToCharArray()).Length - 1];

            var test = new TraficSignPictures_Import();
            var liste = test.Import(Trainingspfad,Testpfad);
            #region csv Import

            //string fileName = @"C:\Users\Joseph\Desktop\TrainingsMuster50.csv"; // Muss angepasst werden
            ////string TestfileName = @"C:\Users\Joseph\Desktop\TestMuster50.csv"; // Muss angepasst werden
            ////string bildTestfileName = @"C:\Users\Joseph\Desktop\0a.csv"; // Muss angepasst werden
            //////string fileName = @"C:\Users\Joseph\Desktop\mnist50.csv"; // Muss angepasst werden
            //////string TestfileName = @"C:\Users\Joseph\Desktop\mnist50.csv"; // Muss angepasst werden
            //////string bildTestfileName = @"C:\Users\Joseph\Desktop\mnist1.csv"; // Muss angepasst werden
            //StreamReader reader = new StreamReader(fileName);
            //string line;
            //string[] splitted;
            //char[] separator = ";".ToCharArray();
            ////char[] separator = ",".ToCharArray();
            //////List<PatternToLearn> pattertolearn = new List<PatternToLearn>();
            //////List<PatternToLearn> pattertotest = new List<PatternToLearn>();
            //////List<PatternToLearn> bildpattertotest = new List<PatternToLearn>();

            //while (!reader.EndOfStream)
            //{
            //    line = reader.ReadLine();
            //    splitted = line.Split(separator);
            //    PatternToLearn muster = new PatternToLearn();
            //    muster.inputvector = new double[784];
            //    muster.targetvector = new double[3];
            //    for (int i = 0; i < splitted.Length; i++)
            //    {
            //        if (i == 0)
            //        {
            //            for (int j = 0; j < muster.targetvector.Length; j++)
            //            {
            //                muster.targetvector[j] = 0d;
            //            }
            //            int temp = Convert.ToInt16(splitted[i]);
            //            muster.targetvector[temp] = 1d;
            //        }
            //        else
            //        {
            //            if (Convert.ToDouble(splitted[i]) > 127)
            //            {
            //                muster.inputvector[i - 1] = 1;
            //            }
            //            else
            //            {
            //                muster.inputvector[i - 1] = 0;
            //            }

            //        }
            //    }
            //    pattertolearn.Add(muster);
            //}
            //reader = new StreamReader(TestfileName);
            //while (!reader.EndOfStream)
            //{
            //    line = reader.ReadLine();
            //    splitted = line.Split(separator);
            //    PatternToLearn muster = new PatternToLearn();
            //    muster.inputvector = new double[784];
            //    muster.targetvector = new double[3];
            //    for (int i = 0; i < splitted.Length; i++)
            //    {
            //        if (i == 0)
            //        {
            //            for (int j = 0; j < muster.targetvector.Length; j++)
            //            {
            //                muster.targetvector[j] = 0d;
            //            }
            //            int temp = Convert.ToInt16(splitted[i]);
            //            muster.targetvector[temp] = 1d;
            //        }
            //        else
            //        {
            //            if (Convert.ToDouble(splitted[i]) > 127)
            //            {
            //                muster.inputvector[i - 1] = 1;
            //            }
            //            else
            //            {
            //                muster.inputvector[i - 1] = 0;
            //            }

            //        }
            //    }
            //    pattertotest.Add(muster);
            //}
            //reader = new StreamReader(bildTestfileName);
            //while (!reader.EndOfStream)
            //{
            //    line = reader.ReadLine();
            //    splitted = line.Split(separator);
            //    PatternToLearn muster = new PatternToLearn();
            //    muster.inputvector = new double[784];
            //    muster.targetvector = new double[3];
            //    for (int i = 0; i < splitted.Length; i++)
            //    {
            //        if (i == 0)
            //        {
            //            for (int j = 0; j < muster.targetvector.Length; j++)
            //            {
            //                muster.targetvector[j] = 0d;
            //            }
            //            int temp = Convert.ToInt16(splitted[i]);
            //            muster.targetvector[temp] = 1d;
            //        }
            //        else
            //        {
            //            if (Convert.ToDouble(splitted[i]) > 127)
            //            {
            //                muster.inputvector[i - 1] = 1;
            //            }
            //            else
            //            {
            //                muster.inputvector[i - 1] = 0;
            //            }

            //        }
            //    }
            //    bildpattertotest.Add(muster);
            //}

            #endregion csv Import

            #region XOR Import

            //List<PatternToLearn> XORpattertolearn = new List<PatternToLearn>();
            //List<PatternToLearn> XORpattertotest = new List<PatternToLearn>();
            //List<PatternToLearn> XORbildpattertotest = new List<PatternToLearn>();

            //XORpattertolearn.Add(new PatternToLearn() {inputvector = new double[2] {0d, 0d}, targetvector = new double[1] {0d}});
            //XORpattertolearn.Add(new PatternToLearn() { inputvector = new double[2] { 1d, 0d }, targetvector = new double[1] { 1d } });
            //XORpattertolearn.Add(new PatternToLearn() { inputvector = new double[2] { 0d, 1d }, targetvector = new double[1] { 1d } });
            //XORpattertolearn.Add(new PatternToLearn() { inputvector = new double[2] { 1d, 1d }, targetvector = new double[1] { 0d } });

            //XORpattertotest = XORpattertolearn;
            //XORbildpattertotest.Add(new PatternToLearn() { inputvector = new double[2] { 1d, 0d }, targetvector = new double[1] { 1d } });

            #endregion XOR Import



            for (int i = 0; i < 5; i++)
            {
                DeepBeliefNetwork DBNN = new DeepBeliefNetwork("backprop");
                DBNN.GreedyLayerWiseTraining(liste[0], liste[1], null, null);
            }

            //for (int i = 0; i < 30; i++)
            //{
            //    Console.Beep();
            //}
            Console.WriteLine("Fertig :D");
            Console.ReadKey();
        }
    }
}