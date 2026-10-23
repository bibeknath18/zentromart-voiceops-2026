import unittest
from datetime import date
from src.core import Retriever, business_days_after, grounded_facts, rows

class TestImprovements(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.r=Retriever()
    def test_holidays_sunday(self):
        self.assertEqual(business_days_after(date(2026,10,17),5),date(2026,10,24))
        self.assertEqual(business_days_after(date(2026,10,17),7),date(2026,10,27))
    def test_hindi_lipstick(self):
        ids=[x['id'] for x in self.r.search('मैंने Gulmohar Matte Lipstick खोल लिया है, क्या अब भी वापस कर सकती हूँ?')]
        self.assertIn('RET-4',ids)
    def test_category_policy(self):
        ids=[x['id'] for x in self.r.search('The Comfi One-piece Swimsuit does not fit. Can I return it?')]
        self.assertIn('RET-5',ids)
    def test_product_lookup(self):
        self.assertIn('"sku": "ELE-0008"', '\n'.join(grounded_facts('Tell me about ELE-0008 warranty')))
    def test_notebook_hindi(self):
        ids=[x['id'] for x in self.r.search('मुझे Pagecraft A5 Ruled Notebook (Pack of 4) लौटाने के लिए कितने दिन मिलेंगे?')]
        self.assertIn('RET-2',ids)
    def test_retrieval_metric(self):
        questions=rows('practice/questions.csv'); vals=[]
        for q in questions:
            gold=[x.strip() for x in q['gold_evidence'].split(';') if x.strip() and '.csv' not in x]
            if gold: vals.append(len(set(gold)&{c['id'] for c in self.r.search(q['question'])})/len(gold))
        self.assertEqual(len(vals),47)
        self.assertGreater(sum(vals)/len(vals),.43)
if __name__=='__main__':unittest.main()
