using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;

namespace DeepBeliefNeuralNetwork
{
    public class TraficSignPictures_Import
    {
        public ConcurrentBag<ConcurrentBag<PatternToLearn>> Import(string Trainingspfad, string Testpfad)
        {
            var returnList = new ConcurrentBag<ConcurrentBag<PatternToLearn>>(); //new List<List<PatternToLearn>>();
            var pattertolearn = new ConcurrentBag<PatternToLearn>();
            var pattertotest = new ConcurrentBag<PatternToLearn>();

            var bilder = new BildLaden();
            System.Threading.Tasks.Parallel.ForEach(Directory.EnumerateDirectories(Trainingspfad), directories =>
            {
                foreach (var file in Directory.EnumerateFiles(directories, "*.jpg"))
                {
                    string splitdirectories = directories.Split(@"\".ToCharArray())[directories.Split(@"\".ToCharArray()).Count() - 1].Remove(0, 3);
                    Bitmap bild = bilder.LoadPicture(file);

                    PatternToLearn muster = new PatternToLearn();
                    muster.inputvector = new double[2304];
                    muster.targetvector = new double[43];
                    int counter = 0;
                    for (int i = 0; i < bild.Height; i++)
                    {
                        for (int j = 0; j < bild.Width; j++)
                        {
                            Color farbe = bild.GetPixel(i, j);
                            if (farbe.R < 127 && farbe.G < 127 && farbe.B < 127)
                            {
                                muster.inputvector[counter] = 0;
                            }
                            else
                            {
                                muster.inputvector[counter] = 1;
                            }
                            counter++;
                        }
                    }
                    muster.targetvector[Int32.Parse(splitdirectories)] = 1;
                    if (muster is null)
                    {
                        Console.WriteLine("test");
                    }
                    pattertolearn.Add(muster);
                }
            }
            );

            List<string[]> compareList = new List<string[]>();
            using (StreamReader reader = new StreamReader(Testpfad + @"\" + @"GT-final_test.csv"))
            {
                string line;
                string[] splitted;
                char[] separator = ";".ToCharArray();

                while (!reader.EndOfStream)
                {
                    string[] compareString = new string[2];
                    line = reader.ReadLine();
                    splitted = line.Split(separator);
                    compareString[0] = splitted[0].Split(".".ToCharArray())[0];
                    compareString[1] = splitted[splitted.Count() - 1];
                    compareList.Add(compareString);
                }
            }

            System.Threading.Tasks.Parallel.ForEach(Directory.EnumerateFiles(Testpfad, "*.jpg"), file =>
            {
                string splittedfile = file.Split(@"\".ToCharArray())[file.Split(@"\".ToCharArray()).Count() - 1].Split(@"-".ToCharArray())[0];
                var matchingvalues = compareList.FirstOrDefault(stringToCheck => stringToCheck[0].Equals(splittedfile));
                Bitmap bild = bilder.LoadPicture(file);
                PatternToLearn muster = new PatternToLearn();
                muster.inputvector = new double[2304];
                muster.targetvector = new double[43];
                int counter = 0;
                for (int i = 0; i < bild.Height; i++)
                {
                    for (int j = 0; j < bild.Width; j++)
                    {
                        Color farbe = bild.GetPixel(i, j);
                        if (farbe.R < 127 && farbe.G < 127 && farbe.B < 127)
                        {
                            muster.inputvector[counter] = 0;
                        }
                        else
                        {
                            muster.inputvector[counter] = 1;
                        }
                        counter++;
                    }
                }
                muster.targetvector[Int32.Parse(matchingvalues[1])] = 1;
                pattertotest.Add(muster);
            });

            returnList.Add(pattertolearn);
            returnList.Add(pattertotest);
            return returnList;
        }
    }
}