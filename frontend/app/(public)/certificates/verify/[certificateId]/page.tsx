import { notFound } from 'next/navigation';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { formatDate } from '@/lib/formatters';
import { ShieldCheck, Award, FileCode, CheckCircle, ExternalLink } from 'lucide-react';
import Link from 'next/link';

const BACKEND_URL = process.env.BACKEND_URL ?? 'http://localhost:8000';

async function verifyCertificate(certId: string) {
  try {
    const res = await fetch(`${BACKEND_URL}/api/v1/certificates/verify/${certId}`, {
      next: { revalidate: 0 },
    });
    if (!res.ok) return null;
    const json = await res.json();
    return json.data;
  } catch {
    return null;
  }
}

export default async function CertificateVerifyPage({
  params,
}: {
  params: Promise<{ certificateId: string }>;
}) {
  const { certificateId } = await params;
  const cert = await verifyCertificate(certificateId);

  if (!cert) {
    notFound();
  }

  const isValid = cert.is_valid !== undefined ? Boolean(cert.is_valid) : cert.status === 'VALID';

  return (
    <div className="mx-auto max-w-4xl px-4 sm:px-6 lg:px-8 py-12 space-y-8">
      <div className="text-center space-y-3">
        <div className="inline-flex items-center gap-2 px-3 py-1 bg-emerald-100 text-emerald-800 text-xs font-semibold rounded-full">
          <ShieldCheck className="h-4 w-4" />
          Cryptographic Proof-of-Impact
        </div>
        <h1 className="text-3xl font-bold text-gray-900">Certificate Verification</h1>
        <p className="text-sm text-gray-500 font-mono">ID: {certificateId}</p>
      </div>

      <Card className="border-emerald-200 bg-emerald-50/20 shadow-md">
        <CardHeader className="flex flex-row items-center justify-between border-b pb-4">
          <div className="flex items-center gap-3">
            <Award className="h-8 w-8 text-emerald-600" />
            <div>
              <CardTitle className="text-lg">Ed25519 Verified Attestation</CardTitle>
              <p className="text-xs text-gray-500">Asymmetric signature mathematically confirmed</p>
            </div>
          </div>
          <Badge variant={isValid ? 'success' : 'danger'}>
            {isValid ? 'VALID' : 'INVALID / REVOKED'}
          </Badge>
        </CardHeader>
        <CardContent className="pt-6 space-y-4 text-sm">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-xs text-gray-500">Project</p>
              <p className="font-semibold text-gray-900">{cert.project_title ?? 'Verified Initiative'}</p>
            </div>
            <div>
              <p className="text-xs text-gray-500">Milestone</p>
              <p className="font-semibold text-gray-900">{cert.milestone_title ?? 'Milestone'}</p>
            </div>
            <div>
              <p className="text-xs text-gray-500">Issuing Organization</p>
              <p className="font-semibold text-gray-900">{cert.organization_name ?? 'Charity'}</p>
            </div>
            <div>
              <p className="text-xs text-gray-500">Issue Date</p>
              <p className="font-semibold text-gray-900">{formatDate(cert.issued_at)}</p>
            </div>
          </div>

          {cert.public_proof && (
            <div className="pt-4 border-t border-gray-200">
              <p className="text-xs font-semibold text-gray-700 mb-2 flex items-center gap-1.5">
                <FileCode className="h-4 w-4 text-blue-600" />
                Raw Cryptographic Proof Envelope (RFC 8785)
              </p>
              <pre className="bg-gray-900 text-gray-100 p-4 rounded text-xs overflow-x-auto font-mono">
                {JSON.stringify(cert.public_proof, null, 2)}
              </pre>
            </div>
          )}

          <div className="pt-4 flex justify-between items-center text-xs text-gray-500 border-t">
            <span>Verified using root JWKS public key discovery</span>
            <Link href="/projects" className="text-blue-600 hover:underline">
              Browse More Projects →
            </Link>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
