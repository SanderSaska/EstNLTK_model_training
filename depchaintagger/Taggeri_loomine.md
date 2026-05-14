<h1 style="color:blue">Praktikum 9</h1>
<h3 style="color:blue">Tekstitöötluse koodi pakendamine märgendajate abil. <br> Osalausestamine</h3>

Käesolevas praktikumis vaatame, kuidas oma tekstitöötluse koodi "pakendada" EstNLTK märgendaja kujule. Lingvistilise analüüsi poole pealt uurime osalausete tuvastajat ja selle kasutusvõimalusi.

## Märgendajate loomine

Selleks, et tekstianalüüsi kood oleks paremini hallatav, taaskasutatav ja teistega jagatav, on mõistlik järgida koodi ülesehitamisel kindlaid printsiipe.
Näiteks tekstianalüüsi sisend ja väljund peaksid olema selgelt defineeritud -- eriti sellisel juhul, kui analüüsil on sõltuvused ning enne analüüsi teostamist tuleb tekitada mitmed teised analüüsid.
Samuti peaks olema võimalik analüüse ja vaheetappe salvestada ning taastada failist (või andmebaasist).
Seetõttu pakubki EstNLTK välja programmeerimisliidese tekstitöötluse koodi "pakendamiseks" -- teksti analüüsi teostab märgendaja (`Tagger`) ning analüüsi tulemuseks on märgenduskiht (`Layer`).
Enamus tekstianalüüsi komponente, millega me seni oleme kokku puutunud (nt teksti lausestaja ja sõnestaja, morfoloogiline analüsaator, nimeüksuste tuvastaja jne), ongi loodud kui `Tagger`-id, mis opereerivad `Layer` objektidega.

Tekstianalüüsi pakendamine `Tagger` ja `Layer` abil tagab, et:

- märgenduskihtide loomisel teostatakse eel- ja järelkontrolli ning tänu sellele järgivad andmed ühtset struktuuri;
- märgendajate konfiguratsiooni ja märgenduskihte saab visualiseerida Jupyter Notebook'is;
- märgenduskihte saab salvestada JSON kujul failidesse / lugeda failidest;
- märgenduskihte saab salvestada vabavaralisse [Postgresql andmebaasi](https://github.com/estnltk/estnltk/tree/main/tutorials/storage) / lugeda andmebaasist. See muutub eriti kasulikuks suurte tekstimassivide töötlemisel;

Selleks, et luua ise märgendajaid ja märgenduskihte, tuleb natukene kursis olla ka Pythoni objekt-orienteeritud poolega -- hea on omada algteadmisi klasside ja objektide loomise kohta, samuti võiks tunda põhimõisteid.
Kui vajad selles osas teadmiste uuendamist, vaata enne edasilugemist üle "Lisamaterjal: Sissejuhatus objekt-orienteeritud programmeerimisse".
Samuti on soovitav kõrvale lahti võtta Moodle's olev "EstNLTK baastarkuste materjal", kus on detailsemalt käsitletud kihtide ja märgendajate loomist.

---

Vaatame nüüd märgendaja loomist lähemalt. Olgu eesmärgiks luua märgendaja, mis tuvastab tekstis tsitaadid ehk jutumärkide vahelised tekstilõigud.
Näidistekst märgendaja katsetamiseks:

```python
text_str = '''
Mees helistab arvutiabi telefonile ja kurdab: "Mu arvutil läks pilt eest ära!"
"Vaadake, kas kõik juhtmed on sees!" ütleb spetsialist.
"Ei saa, pime on!" ütleb hädaline.
"Pange siis tuli põlema."
"Ei saa, elektrit ei ole!"
'''
```

#### Märgenduskihi tekitamine

Sisuliselt toimub märgendaja sees märgenduskihi ehk `Layer` objekti tekitamine.
Kõigepealt vaatame, kuidas käib kihi tekitamine eraldiseisvalt (st ilma märgendajata).
Selleks võtame kasutusele regulaaravaldise, mis tuvastab tekstist jutumärkidega ümbritsetud lõike.
Leiame iga lõigu alguse, lõpu ning tekstisisu:

```python
import re
quotations_regexp = re.compile('("[^"]+")')
for match_obj in list( quotations_regexp.finditer( text_str ) ):
    print( match_obj.start(1),   # lõigu algus
           match_obj.end(1),     # lõigu lõpp
           match_obj.group(1) )  # lõigu sisu (sõne)
```

Edasi loome näidisteksi põhjal uue `Text` objekti ning seejärel ka uue märgenduskihi ehk `Layer` objekti:

```python
from estnltk import Text, Layer
text_obj = Text( text_str )
text_obj
layer = Layer(name='quotations', attributes=(), text_object=text_obj)
layer.meta['allikas'] = 'anekdoot'
```

Kihi nimeks sai `'quotations'` ja atribuute kihil esialgu ei ole. Küll aga andsime kihile edasi info selle kohta, milline on kihile vastav tekst (`text_object=text_obj`). NB! Kui `text_object` jääb määramata, siis saab kihile küll märgendusi lisada, aga pole võimalik kuvada selle tekstilist sisu (st seda, millised märgendused millist tekstifragmenti katavad).

Nagu `Text` objektile, nii on ka kihile võimalik kaasa anda meta-andmeid atribuudi `.meta` abil (mis on sisuliselt sõnastik).

Esialgu on loodud kiht tühi:

```python
layer
```

Kasutame nüüd jälle [finditer](https://docs.python.org/3.11/library/re.html#re.finditer) meetodit, et leida tekstist kõik jutumärkide-vahelised lõigud, võtame välja lõikude asukohad (algus ja lõpp-indeksid) ning lisame kihile `add_annotation` meetodi abil. Tulemuseks ongi märgendustega kiht:

```python
for match_obj in quotations_regexp.finditer( text_obj.text ):
    quotation_start = match_obj.start(1)
    quotation_end   = match_obj.end(1)
    layer.add_annotation( (quotation_start, quotation_end) )
```

```python
layer
```

Rohkem infot märgenduskihtide loomise kohta (kuidas luua eri tüüpi kihte: alamkihte, mitmeseid kihte ja ümbriskihte) leiad _estnltk baastarkuste materjalist_ .

#### Märgendaja loomine

Objekt-orienteeritud disainipõhimõtet järgides peaks ka `Layer` objekt loodama mingi klassi/objekti sees. EstNLTK-s ongi selleks puhuks märgendaja ehk `Tagger` klass.

Märgendaja loomiseks tuleb tekitada `estnltk.taggers.Tagger`-i alamklass ning defineerida:

- klassimuutuja `conf_param`, mis sisaldab märgendaja konfiguratsiooni;
- konstruktor `__init__`, mille ülesanne on algseadistada märgendaja (ja vajadusel laadida sisse tööks vajalikud ressursid);
- `_make_layer_template` meetod, mis tekitab ja tagastab tühja kihi ( nn _kihi malli_ ), kus atribuudid on määratud vastavalt märgendaja algseadistusele;
- `_make_layer` meetod, mille ülesanne on tekitada ( `_make_layer_template` abil) uus märgeduskiht, täita see andmetega ning tagastada;
<div class="alert alert-block alert-warning">
<h4><i>Aga kuidas see kõik kokku käib? ( veel tehnilisem jutt )</i></h4>
<br>
Seni oleme teksti märgendamiseks kasutanud <code>tag</code> meetodit, mille rakendamisel toimub aga "taustal" veel 3 meetodi rakendamine. 
Nendeks meetoditeks on (väljakutsumise järjekorras): <code>make_layer</code> =><code>_make_layer</code> => <code>_make_layer_template</code>. 
Kõik need meetodid panustavad <code>Layer</code> objekti loomisesse.


<code>Tagger</code>-i <code>tag</code> meetodis kutsutakse kihi loomiseks välja meetod <code>make_layer</code> ning kui kihi loomine õnnestub, riputatakse see <code>Text</code> objekti külge.

Ka meetod <code>make_layer</code> (NB! ilma alakriipsuta alguses) ei tegele ise kihi tekitamisega, vaid kontrollib enne kihi tekitamist, kas kõik kihi loomiseks vajalikud sisendkihid on olemas, seejärel kutsub kihi tekitamiseks välja <code>Tagger</code>-i meetodi <code>\_make_layer</code> ning kõige lõpus kontrollib, kas loodud kiht on valiidse struktuuriga (nt sisaldab deklareeritud atribuute).

Meetod <code>\_make_layer</code> kutsub omakorda välja <code>\_make_layer_template</code>-i, et tekitada uus märgendaja algseadistusele vastav kiht, täidab kihi andmetega ning tagastab selle.

Meetodid <code>tag</code> ja <code>make_layer</code> on kõigil märgendajatel ühesugused, neid muuta ega üle defineerida pole tarvis; küll aga tuleb uue <code>Tagger</code>-i loomisel ise implementeerida <code>\_make_layer</code> ja <code>\_make_layer_template</code>.
<br><br>
Aga miks on ikkagi kihi loomine jagatud laiali eri meetodite vahel?
<br><br>
<code>tag</code> ja <code>make_layer</code> eristamine on vajalik selleks, et oleks võimalik luua nii lõppkasutaja jaoks mugavaid <code>Text</code>-iga seotud kihte kui ka nn eraldatud kihte (ingl <i>detached layers</i>), mis pole <code>Text</code>-i alla riputatud. Eraldatud kihte kasutab Postgres andmebaasiliides (kihte saab <code>Text</code>-ist eraldi andmebaasi salvestada) ning neid saab kasutada ka rakendustes, kus kihi loomine on ainult ajutine vahe-etapp -- kui loodud kihilt on informatsioon kätte saadud, pole põhjust seda <code>Text</code>-i külge kinnitada, vaid võib selle ka ära visata (nii hoiab näiteks kokku mäluruumi).
<br>
<code>make_layer</code> ja <code>\_make_layer</code> eristamine võimaldab aga eraldada rutiinsed eel- ja järeltegevused (vajalike sisendkihtide kontroll ja märgenduste valideerimine) otsesest kihi loomisest.
<br>
<code>\_make_layer</code> ja <code>\_make_layer_template</code> võimaldavad eristada kihi malli (milline on kihi struktuur?) ning konkreetsete andmetega täidetud kihti.
Selline eristus on vajalik Postgres andmebaasiliideses: kõigepealt fikseeritakse "kihi malli" abil andmete formaat (luuakse uus tabel), misjärel on võimalik juba konkreetsete andmetega täidetud kihtide salvestamine, mis võib toimuda näiteks paralleeltöötluse võtteid kasutades (st mitu lõime/protsessi võivad parallelselt teostada märgenduskihtide loomist ja salvestamist tabelisse).

</div>
Vaatame nüüd samm-sammult tsitaatide märgendaja defineerimist.

**1)** Impordime vajalikud klassid ning loome `QuotationsTagger`-i, mis on `Tagger`-i alamklass (ülemklassi nimi läheb alamklassi nime järele sulgudesse):

```python
from estnltk import Text, Layer
from estnltk.taggers import Tagger

class QuotationsTagger(Tagger):
    """Tags quotations in Text."""
    conf_param = ['quotations_regexp']
    ...
```

Kui märgendaja vajab oma tööks mingeid ressursse, siis nende nimed (ehk: vastavate isendimuutujate nimed) tuleb panna klassimuutujasse `conf_param`. Käesoleval juhul paneme sinna `quotations_regexp`, kuna meil on plaanis hoida selle nimega muutujas tsitaate eraldavat regulaaravaldist.

**2)** Loome konstruktori, kus on deklareeritud sisendkihid ja väljundkihi nimi, samuti väljundkihi atribuudid ning tööks vajalikud ressursid:

```python
    ...
    def __init__(self, # Väljundkihi nimi muudetavaks:
                       output_layer='quotations'):
        # sisend-väljundkihid ja atribuudid
        self.input_layers = []
        self.output_layer = output_layer
        self.output_attributes = ['length']
        # tööks vajalikud ressursid
        self.quotations_regexp = re.compile('("[^"]+")')
    ...
```

Sisendkihte (`input_layers`) antud juhul pole: regulaaravaldist saame rakendada "toorel tekstil", muid kihte tarvis pole.
Väljundkihi nimi on vaikimisi `'quotations'`, aga seda saab konstruktori parameetriga muuta.
Kui kihi nimi on muudetav, on meil võimalik märgendaja arendust versioneerida, st võime tekitada kihid nimedega `'quotations_v1'`, `'quotations_v2'` jne ning võrrelda neid omavahel.
Isendimuutujas `output_attributes` on loetletud väljundkihi atribuudid; käesolevas näites lisame väljundkihile atribuudi `'length'`, kuhu hakkame salvestama tsitaatide (sõnede) pikkuseid.
Ning lõpuks kompileerime tsitaate eraldava regulaaravaldise ja riputame isendimuutuja `self.quotations_regexp` alla.

**3)** Loome `_make_layer_template` meetodi, mille ülesandeks on luua nn kihi mall (ehk siis tühi märgenduskiht, mis vastab märgendaja algseadistusele):

```python
    def _make_layer_template(self):
        return Layer(name=self.output_layer, attributes=self.output_attributes)
```

Kihi loomisel kasutame muutujaid `self.output_layer` ja `self.output_attributes`, mis annavad märgendaja konfiguratsiooni.
Nii on tagatud, et konfiguratsiooni muutes (nt `output_layer` väärtust muutes) vastaks loodav märgenduskiht konfiguratsioonile.

**4)** Loome `_make_layer` meetodi, mille ülesandeks on luua etteantud `Text` objekti põhjal uus märgenduskiht:

```python
    ...
    def _make_layer(self, text, layers, status):
        # Loome uue kihi
        layer = self._make_layer_template()
        layer.text_object = text
        # Täidame kihi andmetega
        for match in self.quotations_regexp.finditer(text.text):
            quotation_start  = match.start(1)
            quotation_end    = match.end(1)
            annotation       = {'length':quotation_end - quotation_start}
            layer.add_annotation( (quotation_start, quotation_end), annotation )
        # Tagastame loodud kihi
        return layer
```

Meetod `_make_layer` saab argumentideks märgendatava `text`-i (`Text` objekti), olemasolevate kihtide sõnastiku `layers` (võtmeteks `Text` objekti kihinimed ja väärtusteks neile vastavad kihid) ning sõnastiku `status` (sinna võib salvestada metaandmeid kihi loomise protsessi kohta, _a la_ kas loomine õnnestus või kui palju see aega võttis).

Edasi luuakse `_make_layer_template` abil uus (tühi) kiht ning seotakse see ühepoolselt `Text` objektiga.

Nagu eelnevas tsitaatide märgendamise näites, nii tuvastame ka nüüd võimalike tsitaatide asukohad [finditer](https://docs.python.org/3.8/library/re.html#re.finditer) abil.
Lisaks leiame iga tuvastatud tsitaadi pikkuse (sümbolites) ning paneme sõnastikku `annotation` võtme `'length'` alla.
Meetod `add_annotation` võtab 2 sisendargumenti: märgendatav tekstipositsioon `(quotation_start, quotation_end)` ning märgenduse sisu sõnastiku `annotation` kujul -- selle sõnastiku põhjal tekitatakse märgendusele atribuudid.

Töö lõpus tagastame loodud kihi.

Meetodis `_make_layer` meil olemasolevate kihtide sõnastikku `layers` vaja ei läinudki, kuna `self.input_layers = []` ehk märgendaja ei nõudnud sisendkihte.
Aga kui sisendkihid oleks nõutud, tuleks need sõnastikust välja noppida (nt `input_layer_1 = layers[ self.input_layers[0] ]`, `input_layer_2 = layers[ self.input_layers[1] ]` jne) ning seejärel saaks neid kasutada uue kihi tekitamisel.
Oluline on rõhutada, et kõik nõutud kihid tuleb võtta just sõnastikust `layers`, mitte `text`-i alt (nagu seda tehakse tavaliselt). Miks? Märgendajat peab olema võimalik kasutada ka eraldatud kihtide loomiseks ehk siis selliste kihtide loomiseks, millest ükski ei ole veel `text`-i alla riputatud.
Paneme nüüd `QuotationsTagger`-i koodi kokku üheks klassiks:

```python
import re

from estnltk import Text, Layer
from estnltk.taggers import Tagger

class QuotationsTagger(Tagger):
    """Tags quotations in Text."""
    conf_param = ['quotations_regexp']

    def __init__(self, # Väljundkihi nimi muudetavaks:
                       output_layer='quotations'):
        # sisend-väljundkihid ja atribuudid
        self.input_layers = []
        self.output_layer = output_layer
        self.output_attributes = ['length']
        # tööks vajalikud ressursid
        self.quotations_regexp = re.compile('("[^"]+")')

    def _make_layer_template(self):
        return Layer(name=self.output_layer, attributes=self.output_attributes)

    def _make_layer(self, text, layers, status):
        # Loome uue kihi
        layer = self._make_layer_template()
        layer.text_object = text
        # Täidame kihi andmetega
        for match in self.quotations_regexp.finditer(text.text):
            quotation_start  = match.start(1)
            quotation_end    = match.end(1)
            annotation       = {'length':quotation_end - quotation_start}
            layer.add_annotation( (quotation_start, quotation_end), annotation )
        # Tagastame loodud kihi
        return layer
```

Katsetame. Loome uue märgendaja:

```python
quotes_tagger = QuotationsTagger()
quotes_tagger
```

Nüüd, kui märgendaja on olemas, saab seda teksti märgendamiseks välja kutsuda kahel viisil.

**1)** Meetod `make_layer` tekitab uue kihi ja tagastab selle, aga ei riputa kihti `text`-i külge -- tegemist on nn eraldatud kihiga:

````python
text = Text(text_str)
quotes_tagger.make_layer(text)
```python
# Tekstil pole veel ühtegi kiht
text.layers
````

**2)** Meetod `tag` tekitab uue kihi ning tagastab `Text` objekti-i, kuhu kiht on külge riputatud:

```python
quotes_tagger.tag(text)
```

```python
# uurime külge riputatud kihti
text.quotations.display()
```

<div class="alert alert-block alert-warning">
<h4><i>Märgendajad vs ümbermärgendajad</i></h4>
<ul>
<li>Kui <code>Tagger</code> on mõeldud uue kihi tekitamiseks, siis lisaks on olemas veel  <code>Retagger</code>, mis tegeleb olemasoleva kihi muutmise või parandamisega. Näiteks on <code>Retagger</code>-id morfoloogilise analüüsi kihti muutvad <code>VabamorfDisambiguator</code> (lausepõhine ühestaja) ja <code>UserDictTagger</code> (kasutajasõnastikupõhine parandaja). <code>Retagger</code> liides kasutab samu atribuute, mis <code>Tagger</code>, aga meetodite nimed on teistsugused: <code>_make_layer(...)</code> asemel on <code>_change_layer(...)</code>, <code>make_layer(...)</code> asemel <code>change_layer(...)</code> ning <code>tag(...)</code> asemel <code>retag(...)</code>;</li>
<li>Detailsemat informatsiooni ja lisanäiteid märgendajate / ümbermärgendajate tekitamise kohta leiad estnltk baastarkuste materjalist.</li>
</ul>
</div>
