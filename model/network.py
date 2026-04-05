# model/network.py
from brian2 import *
from .neuron import get_izhikevich_eqs, get_state_params
import inout

def create_network():
    n_input = inout.INPUT_NEURONS
    n_output = inout.OUTPUT_NEURONS
    
    # حساب عدد الـ Hidden Neurons بناءً على معادلتك
    n_hidden = int((2.0 / 3.0) * n_input + n_output)
    
    print(f"Network Topology: {n_input} Input -> {n_hidden} Hidden -> {n_output} Output")
    
    state_params = get_state_params(inout.NEURON_STATE)
    print(f"Selected Neuron State: {inout.NEURON_STATE} - {state_params['name']}")

    eqs, threshold, reset = get_izhikevich_eqs()

    # طبقة الإدخال (Poisson Generator لإنتاج Spikes بناءً على البيانات)
    inp_group = PoissonGroup(n_input, rates=0*Hz)

    # دالة مساعدة لإنشاء الخلايا وضبط الـ Parameters لها
    def make_izhikevich_group(n):
        G = NeuronGroup(n, eqs, threshold=threshold, reset=reset, method='rk4') # rk4 لأعلى دقة تفاضل
        G.v = state_params['c']
        G.u = G.v * state_params['b']
        G.a = state_params['a']
        G.b = state_params['b']
        G.c = state_params['c']
        G.d = state_params['d']
        G.I = 0
        return G

    hidden_group = make_izhikevich_group(n_hidden)
    output_group = make_izhikevich_group(n_output)

   # تجهيز معادلات التشابك العصبي للـ STDP الموجه بالمكافأة (RSTDP)
    tau_pre, tau_post = 20*ms, 20*ms
    A_pre, A_post = 0.01, -0.012

    # ====== الحل هنا: نجمع المتغيرات في قاموس (Dictionary) ======
    syn_namespace = {
        'tau_pre': tau_pre, 
        'tau_post': tau_post, 
        'A_pre': A_pre, 
        'A_post': A_post
    }

    syn_eqs = '''
    w : 1
    delta_w : 1  # يسجل التغيير المحتمل في الوزن لحين تقييم المكافأة
    dapre/dt = -apre/tau_pre : 1 (event-driven)
    dapost/dt = -apost/tau_post : 1 (event-driven)
    '''
    on_pre = '''
    v_post += w * 10  # تحفيز قوي لنقل الإشارة
    apre += A_pre
    delta_w += apost
    '''
    on_post = '''
    apost += A_post
    delta_w += apre
    '''

    # ====== ونمرر الـ namespace هنا ======
    S_in_hid = Synapses(inp_group, hidden_group, syn_eqs, on_pre=on_pre, on_post=on_post, method='rk4', namespace=syn_namespace)
    S_in_hid.connect(p=1.0) # Fully connected
    S_in_hid.w = 'rand() * 2' # Initial weights
    S_in_hid.delta_w = 0

    S_hid_out = Synapses(hidden_group, output_group, syn_eqs, on_pre=on_pre, on_post=on_post, method='rk4', namespace=syn_namespace)
    S_hid_out.connect(p=1.0)
    S_hid_out.w = 'rand() * 2'
    S_hid_out.delta_w = 0

    # إضافة روابط استرجاعية (Recurrent) للطبقة المخفية لتكوين "ذاكرة سياق"
    S_hid_hid = Synapses(hidden_group, hidden_group, syn_eqs, on_pre=on_pre, on_post=on_post, method='rk4', namespace=syn_namespace)
    
    # لا نربط كل النيورونات ببعضها (لكي لا تحدث صرع/Epilepsy للشبكة)، نربط نسبة معينة مثلاً 20%
    S_hid_hid.connect(p=0.2, condition='i != j') # i != j تمنع النيورون من إرسال إشارة لنفسه مباشرة
    S_hid_hid.w = 'rand() * 0.5' # أوزان ضعيفة لتعمل كـ "همس" في الخلفية
    S_hid_hid.delta_w = 0
    
    # طبعاً ستحتاج لعمل return لـ S_hid_hid مع باقي الروابط

    return inp_group, hidden_group, output_group, S_in_hid, S_hid_out