## 2.2 Märgendajad / Ümbermärgendajad

### 2.2.1 Tagger (kihi looja)

Märgendaja abil luuakse uusi kihte. Märgendaja tegemine samm-haaval:

1. Loo `estnltk.taggers.Tagger` klassi alamklass;
2. Lisa kõik märgendaja konfiguratsiooni parameetrid klassimuutujasse `conf_param` (muutuja tüüp `Sequence[str]`);
   Ainult selles loendis olevaid parameetreid tohib märgendaja alla salvestada. Kui parameetri nime ees on alakriips (`_`), siis loetakse see sisemiseks parameetriks ning märgendaja konfiguratsiooni kuvamisel seda ei näidata;
3. Lisa väljundkihi nimi klassimuutujasse `output_layer` (tüüp `str`);
4. Lisa kõik väljundkihi atribuutide nimed muutujasse `output_attributes` (tüüp `Sequence[str]`);
5. Lisa kõik kihid, mida on vaja väljundkihi loomiseks, muutujasse `input_layers` (tüüp `Sequence[str]`);
6. Loo konstruktor `__init__`, kus pannakse lõplikult paika parameetrite `conf_param`, `output_layer`, `output_attributes`, `input_layers` ning teiste konfiguratsiooniparameetrite väärtused. Märgendaja konfiguratsioon peaks olema täielikult määratud konstruktoris, väljaspool konstruktorit selle muutmist toimuda ei tohiks. (Kui kasutajal on vaja muuta konfiguratsiooni, peaks ta tekitada uue märgendaja uue konfiguratsiooniga);
7. Loo meetod `_make_layer_template(self) -> Layer`, mis tekitab tühja eraldatud kihi, kus on kõik kihi atribuudid seadistatud vastavalt märgendaja konfiguratsioonile;
8. Loo meetod `_make_layer(self, raw_text: str, layers: Mapping[str, Layer], status: dict=None) -> Layer`, milles tekitatakse uus `Layer` objekt (soovitatavalt `_make_layer_template()` abil), täidetakse see andmetega ning tagastatakse;

Lihtsustatud näide: loome märgendaja, mis märgendab toiduretseptides koguseid (nt 1 tl , 200 g , 2 dl ):

```python
from estnltk import Text, Layer
from estnltk.taggers import Tagger

class QuantityTokensTagger(Tagger):
"""Tags tokens that make up quantity expressions."""
conf_param = ['quantity_lemmas']

    def __init__(self, # output_layer name can be changed:
                       output_layer='quantity_tokens',
                       # input layer name can be changed:
                       input_morph_analysis_layer='morph_analysis',
                       # quantity lemmas can be changed:
                       quantity_lemmas=['tk', 'tl', 'dl', 'kg', 'g']):
        # Set input/output layers
        self.input_layers = [input_morph_analysis_layer]
        self.output_layer = output_layer
        self.output_attributes = ['token_type']
        # Set other configuration parameters
        self.quantity_lemmas = set(quantity_lemmas)

    def _make_layer_template(self):
        # Create new detached layer debased on the configuration
        return Layer(name=self.output_layer, attributes=self.output_attributes, text_object=None)

    def _make_layer(self, text, layers, status):
        # Create new layer based on the configuration
        layer = self._make_layer_template()
        # Assign the Text object
        layer.text_object = text
        for span in layers[ self.input_layers[0] ]: # Iterate over 'morph_analysis' (first input layer)
            for annotation in span.annotations:
                if annotation['lemma'] in self.quantity_lemmas:
                    # Mark units
                    layer.add_annotation(span.base_span, token_type='UNIT')
                    break
                if annotation['lemma'].replace('.','',1).isdigit():
                    # Mark numbers
                    layer.add_annotation(span.base_span, token_type='NUMBER')
                    break
        # Return created layer
        return layer
```

Testime märgendajat:

```python
quantities_tagger = QuantityTokensTagger()
quantities_tagger
```

```
Tagger
Tags tokens that make up quantity expressions.
name output layer output attributes input layers
QuantityTokensTagger quantity_tokens ('token_type',) ('morph_analysis',)
Configuration
quantity_lemmas {'tk', 'kg', 'g', 'dl', 'tl'}
```

# Loome näidisteksti koos vajaminevate sisendkihtidega

```python
text = Text('''
200 g tumedat 70% šokolaadi (Fairtrade)
200 g võid
100 g hakitud kreeka pähkleid
0.5 tl soola
0.5 tl vanilliekstrakti
''')
text.tag_layer('morph_analysis')

# Rakendame märgendajat

quantities_tagger.tag( text )

# Visualiseerime tulemused

text.quantity_tokens.display()
display( text.quantity_tokens )
```

```
200 g tumedat 70% šokolaadi (Fairtrade)
200 g võid
100 g hakitud kreeka pähkleid
0.5 tl soola
0.5 tl vanilliekstrakti
Layer
layer name attributes parent enveloping ambiguous span count
quantity_tokens token_type None None False 10
text token_type
200 NUMBER
g UNIT
200 NUMBER
g UNIT
100 NUMBER
g UNIT
0.5 NUMBER
tl UNIT
0.5 NUMBER
tl UNIT
```

Rohkem detaile

- Sisuliselt on `Tagger`-il neli meetodit, mis haldavad kihi loomist:
  - `tag(text: Text, status: dict)` -- luuakse ja lisatakse kiht etteantud `Text` objektile. Lõppkasutajale mõeldud meetod, mida märgendajate arendajad reeglina muuta / üle kirjutada ei tohiks;
  - `make_layer(text: Text, layers: MutableMapping[str, Layer], status: dict)` -- luuakse ja tagastatakse loodud kiht ilma seda `Text` objektiga sidumata. Vajalik Postgres andmebaasiliidesele: meetodi abil saab tekitada eraldatud kihi (detached layer), mis on andmebaasis `Text` objektist eraldi. Märgendajate arendajad reeglina seda meetodit muuta / üle kirjutada ei tohiks;
  - `\_make_layer(text: Text, layers: MutableMapping[str, Layer] = None, status: dict = None)` -- märgenduskihi loomine ja tagastamine. Märgendajate arendajad peaksid implementeerima selle meetodi;
  - `\_make_layer_template()` -- konfiguratsioonile vastava tühja märgenduskihi loomine ja tagastamine. Reeglina peaks meetod `\_make_layer()` teostama kihi loomist meetodi `\_make_layer_template()` abil, seega tuleb ka see meetod arendajatel implementeerida;
- Kihi loomise staatus: kihi loomise meetoditele saab parameetriks anda sõnastiku status, kuhu märgendaja võib sõnumeid salvestada (nt selle kohta, kas kihi loomine õnnestus);
- Valideerimine: meetodis `make_layer(...)` toimub märgendaja sisendi- ja väljundi valideerimine. Enne kihi loomist kontrollitakse, et on olemas kõik vajalikud sisendkihid (`input_layers`) ning pärast kihi loomist kontrollitakse, et väljundkiht (`output_layer`) on olemas ning et kõigil märgendustel on väljundatribuudid (`output_attributes`);
- Muudetavad kihtide nimed: on soovitatav, et sisend- ja väljundkihtide nimed oleksid märgendajas muudetavad: 1) konstruktori parameetrite abil peaks olema võimalik muuta `input_layers` ja `output_layer` väärtuseid; 2) meetodis \_make_layer(...) peaks eksplitsiitsete kihinimede asemel kasutada kihinimede muutujaid `input_layers` ja `output_layer`. Kui kihinimed on muudetavad, on võimalik võrrelda märgendaja eri versioonide väljundeid;

🔗 Märgendaja loomise kohta vt veel: https://github.com/estnltk/estnltk/blob/main/tutorials/taggers/base_tagger.ipynb
