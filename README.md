# RHpapercode

Primality test is done using inbuilt gmpy2 library, which can be installed using:<br/>
```pip install gmpy2```<br/>

Row reduction in finite field is implemented using an external library called "galois"- a performant NumPy extension for Galois fields and their applications,
which can be can be installed in python3 using ```pip install galois```<br/>

The parameters: $a = 1938$, $b= 31$ generate functions of domain sizes upto 512 or 9-bits. We used the same parameters to generate few example primes that implement functions of the form: 
 ```math
 \begin{equation}
  f(x) =
    \begin{cases}
      ~1 & \text{if $x\geq t$ for threshold t}\\
      0 & \text{otherwise}
    \end{cases}       
\end{equation}
 ```
```
1.p=1355076133667726000417315353609417586606243353926509708629607443890914136735097114671952843635668617386783478279206035797241889819595013909267752192569368438009178097302214384100685640545037756242510244550595482403155309719817823839047721677336790399278341159271648099778050663757988545397129511816651426356835218375133041795508675876661217459441832876122223663872750595398143644849145704620189993519806692040389964139370650663542200860589488824453395993519897418600186874014256365221967265669354453376497698724501391783331192098536207059509026696143648631220885033394519269002005654032573937311239272339981740693132010119855141098177543904086425724930392394838950282083734784982806750183974037037262192084283733875560881713790917363737483072187571094733248799279816549943426066671695886112607314932777312231133005300345483202530107447465274938711919821432022321032556901393725018391864386865786109978019491918762088196210952551713168738366178846764612121385650197334906341620446119078562603255588186752466444812634875218782782842296660458379396049076198744653185202866650447441749143705884847744191958793102032351805432329066280370457874123368721507437438586853778011253124474325691276760846290540673319096611048391815017521793091016149661544084613627404236571027280313580192070814495023123448423143276783258552429611169150081197250425835544281209597090149279644922200885154711935411236632714572556120028097889499936820926446700827953609495462483297833713246401037447428130732929672058495269049427363485647183639138371634298407651476955890374892407461399041520780574443320137028545795062992415922918926095115589289988651499317037530753582101081358329184726531070624603103743450574445262999045484590850372777619728534348172497000970631153012892128826427971478706338814495965755883611446253211344370561564843277970656613330239816815602808908435327864206939963677898661861420648740980850236634205157644201299386323028161763845546796098072122471749935136149212290709852405461833054949605392409671934486806952074332943002690302915434619368519922039254954137041296913639242810114004934902817792459512803453582025801234374190850218743057
```

``` 2.p=1445929899877112985326739779007042144567746613421808498601819345453354045947261967882175039361677317577814960718172629110533644036744739359298953260169136488859319916260104924189115122043608540189000463596061197229351602361187899137415560790289389408064918314037800369963353228726462064977635781041201124254974987001187632004154466605080324993312064316749714587285989168997562993656659948186292388966468445477344949713941146473042938604300479492095974255672239355219273661213547673710678210535840735648344622654512082428703745751187232712854211386964184410846300338537930426897205331670108274435129197530366063714225254905005445353275427425519386639340494889734099009006889325295980584850669295605889665179156308197299141622745919243839139375015509893201561677834889157225867916501087230940234222700435424856614943571672646899781191397036557667044166339786405941592618585996917833879557558803600675043628684886224372231125543407664638518354242386823396142491384042070394339439633735061629946063426508458028330216611415105174190202611266679823707370773371608074323168104848569004954514306178804999298212391045374157117551160282617424821395889755543653318606850119039073679765718602789396301878790403869733911990431308259393556067192451605839625872054825372077525262052039155063316721916429686187212165630861882877554181925139258942124117885793956809229786566673272438455574397771593345478173547679824018411441649039130374745333562508541456198733991115989635313326533508073795247322407490444566271031563621316094924659204265177809029475800992601490447683661724305360012163685320127495699389172706527416090952117445078535067487929552345974871849121034468622421887450591490490421748621565744794009567876864174571479461264017585400917432669958761520136116529297578686450444140435957295581172463568440398378857825385167251347972515814791970340434663134230367885644088982534596011447395553183661142957054414506352003981474323999612730830608853420914490700889076805700106846189616831895168132085474700257784051361138175118871120410818241740886306601375417204673063230408518919096092313423911361165699859741240166780668040485637969679304397
```
 
 ```
 3.p=1445929899877112985326739779007042144567746613421808498601819345453354045947261967882175039361677317577814960718172629110533644036744739359298953260169136488859319916260104924189115122043608540189000463596061197229351602361187899137415560790289389408064918314037800369963353228726462064977635781041201124254974987001187632004154466605080324993312064316749714587285989168997562993656659948186292388966468445477344949713941146473042938604300479492095974255672239355219273661213547673710678210535840735648344622654512082428703745751187232712854211386964184410846300338537930426897205331670108274435129197530366063714225254905005445353275427425519386639340494889734099009006889325295980584850669295605889665179156308197299141622745919243839139375015509893201561677834889157225867916501087230940234222700435424856614943571672646899781191397036557667044166339786405941592618585996917833879557558803600675043628684886224372231125543407664638518354242386823396142491384042070394339439633735061629946063426508458028330216611415105174190202611266679823707370773371608074323168104848569004954514306178804999298212391045374157117551160282617424821395889755543653318606850119039073679765718602789396301878790403869733911990431308259393556067192451605839625872054825372077525262052039155063316721916429686187212165630861882877554181925139258942124117885793956809229786566673272438455574397771593345478173547679824018411441649039130374745333562508541456198733991115989635313326533508073795247322407490444566271031563621316094924659204265177809029475800992601490447683661724305360012163685320127495699389172706527416090952117445078535067487929552345974871849121034468622421887450591490490421748621565744794009567876864174571479461264017585400917432669958761520136116529297578686450444140435957295581172463568440398378857825385167251347972515814791970340434663134230367885644088982534596011447395553183661142957054414506352003981474323999612730830608853420914490700889076805700106846189616831895168132085474700257784051361138175118871120410818241740886306601375417204673063230408518919096092313423911361165699859741240166780668040485637969679304397
 ```

```
4.p=1640309185919357366402558999907871945193168934956456471649227625795936669354341201567250796434434470293482714078246710824400040970040609783822394367258262420451966760998578527152981395710275844147018743597108691352949611077333839759565788684183724928346235190777757248609137599339588575940427074009428619688265454407326536917806011737679966945640269963256182629582073909213941814375775523217443394951353419275440455711420745015893139544198933725622718388337831057783572305556752743839874107216538563546602338295928715447322780091367327467168552459907234376121279154895471865353778296694825273020481921478436500600623569377012928183128288753999866553972284477505383054736565945801451320033925377802032903491946893865030981431000229487678754490871691867486757896448972091447602532653823884256123001713703754959415509254730361464819698650424590763452870654583884260273729263588405206747938875581858857666563788635077264353292146392300312977831508879480571975755978849119986222902640050132261896439187041635268507857494299981636196010809982779166202243747891387631362148752962502196418913958550609697418869269915213474593770588924206570382535797981561829372896185113833669856308428861289887648276656308694719967614415870553263116695061697021900485937208908652293474980255186819446927752981530331892714177005348878394410068830567885264139806462641024237439036088097339705120491009324858567404860683326669029770085349662013339328457810538222495188270531329701242241648414902208947860827111796347417921454066575510006987993347805162427333236024254248291010515269788356194254622678351157567005528158988982645818821603657263731692452564855330953823055253197605878420687110557954197107331129359578644315284077855489739353202115963551431709935494849674418016069751892858184007816866007498095050747895451403070446574047262211476002291185239528452854042450943040928011012394941525633587190179546946740358243665613029004568661213402520138777798402141445174585591803626556107568932471500914038560593641654723336351374945093203379386139648271272904188862191255625274168259645427077135762573180764552270534826180057817963735743269196697914584349033
```
```
5.p=292483522932855437916789597138036960153656025696173450368713690984715510740052310069257216193078685238796246843442126922353956056102651078245150055936383690482213041389171875603942256624781823753288362625710152259508805306689293781539021075227865220831386852895394498408641973407414789908648236229122855259135553220145332394905012456512066887794206751766762477540315348218125989735363490819324693152718809929318733527772779328160298122493130973469429326149488107335867351363430909970807829046927817326020841507433307819586171509236702313995124213015924424396077629244265553694807930574757056386517473471627773492868801129447305896787453715332226570948925179925249894983191292595721509597702316391607222753285308321035892097054560002675339233336792495456882253867494649202082032151653120780440555515176039230606734127741364523112157860692807861701333403693863173287284796435364676516122354521901106287430433176645792356173461275606358470226988563924397250498988799266815776107495147870706069066474073401842794002769176707107304816533574346000831218909559150013870500099816274724591646961495407423231821815773456180526204258315544698934626425913444118288597036931058297722800528238902196833616284139324066295871671695503465963271687774920953182146121161846701793966076173337423688416217704510622213661187620627979717903064678477843309273101195092755401861283978932930169410823135132751078777848564975946262591720709642009610921557468521612291470206201887487905870774992980423157914060338640319936519065670983354319334955718639347000600128411371952164891800294224521850065057021615408786293068168327296061555931324032408131604162990572509442496313042508006196520608749022144859210728297886358467686585364922539540159875063362668003810223596156642247068062667470710120436146742146202651347339359949127352636558254584848308893842182584857789401407503067575774371935038398461739159206521394915463497857940937075650830137665913738362149011651527853915695153215870897329821282877075150073293615731090870272030296264334165517484990547747542772905698925747589591088326119433092648748249928288816326390310357991149378751469444421540640129953
```

We can generate many such primes that accomodate the required patterns of residues and non-residues, within the required parameter lengths. 
