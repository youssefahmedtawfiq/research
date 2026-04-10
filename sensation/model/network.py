# model/network.py
from random import seed
from brian2 import *
from .neuron import get_izhikevich_eqs, get_state_params
import inout
import numpy as np
np.random.seed(42)
seed(42)

def create_network(spike_indices=None, spike_times=None):
    n_input = inout.INPUT_NEURONS
    n_output = inout.OUTPUT_NEURONS
    n_hidden = int((2.0 / 3.0) * n_input + n_output)

    eqs, threshold, reset = get_izhikevich_eqs()
    state_params = get_state_params(inout.NEURON_STATE)

    # التعديل: استخدام SpikeGeneratorGroup بدلاً من PoissonGroup
    if spike_indices is not None and spike_times is not None:
        inp_group = SpikeGeneratorGroup(n_input, spike_indices, spike_times)
    else:
        # مجموعة فارغة في حالة البناء الأولي
        inp_group = SpikeGeneratorGroup(n_input, [], [] * ms)

    # ... بقية تعريف الـ hidden_group و output_group و Synapses كما هي في كودك الأصلي ...
    # (تأكد من إرجاع S_in_hid و S_hid_out)
    
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
    delta_w : 1 
    dapre/dt = -apre/tau_pre : 1 (event-driven)
    dapost/dt = -apost/tau_post : 1 (event-driven)
    '''
  # في الجزء الخاص بـ on_pre
    on_pre = '''
    v_post += w * 30 
    apre += A_pre
    delta_w += apost
    '''
    on_post = '''
    apost += A_post
    delta_w += apre
    '''
# ====== ونمرر الـ namespace هنا ======
    S_in_hid = Synapses(inp_group, hidden_group, syn_eqs, on_pre=on_pre, on_post=on_post, method='rk4', namespace=syn_namespace)
    S_in_hid.connect(p=0.6) # Fully connected
    S_in_hid.delay = '1*ms + rand() * 4*ms' # 🌟 إضافة تأخير زمني عشوائي
    S_in_hid.w = '1.0 + rand() * 2' # 🌟 رفع الأوزان المبدئية
    S_in_hid.delta_w = 0

    S_hid_out = Synapses(hidden_group, output_group, syn_eqs, on_pre=on_pre, on_post=on_post, method='rk4', namespace=syn_namespace)
    S_hid_out.connect(p=0.6)
    S_hid_out.delay = '1*ms + rand() * 4*ms' # 🌟 إضافة تأخير زمني عشوائي
    S_hid_out.w = '1.0 + rand() * 2' # 🌟 رفع الأوزان المبدئية
    S_hid_out.delta_w = 0

    # إضافة روابط استرجاعية (Recurrent) للطبقة المخفية لتكوين "ذاكرة سياق"
    S_hid_hid = Synapses(hidden_group, hidden_group, syn_eqs, on_pre=on_pre, on_post=on_post, method='rk4', namespace=syn_namespace)
    S_hid_hid.connect(p=0.2, condition='i != j')
    S_hid_hid.delay = '1*ms + rand() * 3*ms' # 🌟 تأخير للروابط الداخلية
    S_hid_hid.w = 'rand() * 0.5'
    S_hid_hid.delta_w = 0

    # ⚠️ ملحوظة هامة: في الكود القديم بتاعك كان في سطرين تحت بيمسحوا الأوزان دي (S_in_hid.w = 'rand() * 0.5')
    # لو لسه موجودين عندك امسحهم تماماً، واكتفي بالكود اللي فوق ده.
    
    # طبعاً ستحتاج لعمل return لـ S_hid_hid مع باقي الروابط

    return inp_group, hidden_group, output_group, S_in_hid, S_hid_out, S_hid_hid
